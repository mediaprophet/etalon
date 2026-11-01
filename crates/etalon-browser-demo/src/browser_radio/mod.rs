mod browser;
mod capabilities;
mod value;
pub use capabilities::radio_capabilities;

use browser::*;
use js_sys::{Array, DataView, Object, Promise, Reflect, Uint8Array};
use serde_json::{json, Value};
use std::{cell::RefCell, rc::Rc};
use value::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

type Listener = Closure<dyn FnMut(JsValue)>;
thread_local! { static OWNER: RefCell<Option<std::rc::Weak<RefCell<State>>>> = const { RefCell::new(None) }; }
fn claim_owner(state: &Rc<RefCell<State>>) -> Result<(), String> {
    OWNER.with(|owner| {
        if let Some(other) = owner.borrow().as_ref().and_then(|o| o.upgrade()) {
            if !Rc::ptr_eq(&other, state) && (other.borrow().busy || other.borrow().connected()) {
                return Err("Another Bluetooth session owns the active connection".into());
            }
            if !Rc::ptr_eq(&other, state) {
                other.borrow_mut().cleanup();
            }
        }
        *owner.borrow_mut() = Some(Rc::downgrade(state));
        Ok(())
    })
}
fn gatt_reused(gatt: &JsValue) -> bool {
    OWNER.with(|owner| {
        owner
            .borrow()
            .as_ref()
            .and_then(|o| o.upgrade())
            .and_then(|s| s.borrow().device.as_ref().and_then(|d| get(d, "gatt").ok()))
            .is_some_and(|current| Object::is(&current, gatt))
    })
}
#[derive(Default)]
struct State {
    epoch: u64,
    busy: bool,
    device: Option<JsValue>,
    characteristic: Option<JsValue>,
    listener: Option<Listener>,
    disconnected: Option<Listener>,
    subscribed: bool,
    events: Events,
}
impl State {
    fn connected(&self) -> bool {
        self.device
            .as_ref()
            .and_then(|d| get(d, "gatt").ok())
            .and_then(|g| boolean(&g, "connected"))
            .unwrap_or(false)
    }
    fn cleanup(&mut self) {
        if let (Some(c), Some(l)) = (&self.characteristic, &self.listener) {
            let _ = call(
                c,
                "removeEventListener",
                &["characteristicvaluechanged".into(), l.as_ref().clone()],
            );
        }
        if let (Some(d), Some(l)) = (&self.device, &self.disconnected) {
            let _ = call(
                d,
                "removeEventListener",
                &["gattserverdisconnected".into(), l.as_ref().clone()],
            );
        }
        if let Some(d) = &self.device {
            if let Ok(g) = get(d, "gatt") {
                let _ = call(&g, "disconnect", &[]);
            }
        }
        self.characteristic = None;
        self.listener = None;
        self.disconnected = None;
        self.device = None;
        self.subscribed = false;
        self.busy = false;
    }
}
struct Operation {
    state: Rc<RefCell<State>>,
    epoch: u64,
}
impl Operation {
    fn begin(state: &Rc<RefCell<State>>) -> Result<Self, String> {
        let mut s = state.borrow_mut();
        if s.busy {
            return Err("Bluetooth operation already in progress".into());
        }
        s.busy = true;
        Ok(Self {
            state: Rc::clone(state),
            epoch: s.epoch,
        })
    }
    fn check(&self) -> Result<(), String> {
        if self.state.borrow().epoch != self.epoch {
            Err("Bluetooth operation cancelled or device disconnected".into())
        } else {
            Ok(())
        }
    }
    fn characteristic(&self) -> Result<JsValue, String> {
        let s = self.state.borrow();
        if !s.connected() {
            return Err("Bluetooth device is disconnected".into());
        }
        s.characteristic
            .clone()
            .ok_or_else(|| "No characteristic selected".into())
    }
}
impl Drop for Operation {
    fn drop(&mut self) {
        let mut s = self.state.borrow_mut();
        if s.epoch == self.epoch {
            s.busy = false;
        }
    }
}
fn response(state: &Rc<RefCell<State>>, result: Result<Value, String>) -> String {
    let s = state.borrow();
    let mut v = match result {
        Ok(v) => v,
        Err(e) => json!({"ok":false,"error":e}),
    };
    v["schema"] = json!("etalon.browser-radio.v1");
    v["source_revision"] =
        json!(option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"));
    v["library_version"] = json!(env!("CARGO_PKG_VERSION"));
    v["connected"] = json!(s.connected());
    v["subscribed"] = json!(s.subscribed && s.connected());
    v.to_string()
}
fn packet(v: &JsValue) -> Result<Packet, String> {
    let view = v
        .dyn_ref::<DataView>()
        .ok_or("Characteristic value is not a DataView")?;
    let len = view.byte_length();
    if len > MAX_VALUE {
        return Err("Characteristic exceeds 512 bytes".into());
    }
    let mut p = Packet {
        len,
        timestamp_ms: js_sys::Date::now(),
        ..Packet::default()
    };
    for i in 0..len {
        p.bytes[i] = view.get_uint8(i);
    }
    Ok(p)
}
fn properties(c: &JsValue) -> Value {
    let Ok(p) = get(c, "properties") else {
        return json!({});
    };
    json!({"read":boolean(&p,"read").unwrap_or(false),"write":boolean(&p,"write").unwrap_or(false),"write_without_response":boolean(&p,"writeWithoutResponse").unwrap_or(false),"notify":boolean(&p,"notify").unwrap_or(false),"indicate":boolean(&p,"indicate").unwrap_or(false)})
}

#[wasm_bindgen]
pub struct BrowserBluetooth {
    state: Rc<RefCell<State>>,
}
#[wasm_bindgen]
impl BrowserBluetooth {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(State::default())),
        }
    }
    pub fn connect(&self, service_uuid: String, characteristic_uuid: String) -> Promise {
        let state = Rc::clone(&self.state);
        let setup = (|| -> Result<(Operation, JsValue, String, String), String> {
            let service = uuid(&service_uuid)?;
            let characteristic = uuid(&characteristic_uuid)?;
            claim_owner(&state)?;
            {
                let mut s = state.borrow_mut();
                if s.busy {
                    return Err("Bluetooth operation already in progress".into());
                }
                s.epoch = s.epoch.wrapping_add(1);
                s.cleanup();
                s.events = Events::default();
            }
            let op = Operation::begin(&state)?;
            let bt = get(&navigator()?, "bluetooth")?;
            let options = Object::new();
            let filter = Object::new();
            let services = Array::new();
            services.push(&service.clone().into());
            Reflect::set(&filter, &"services".into(), &services).map_err(error)?;
            let filters = Array::new();
            filters.push(&filter);
            Reflect::set(&options, &"filters".into(), &filters).map_err(error)?;
            // Invoke the chooser before scheduling the asynchronous connection work.
            let selected = call(&bt, "requestDevice", &[options.into()])?;
            Ok((op, selected, service, characteristic))
        })();
        let epoch = setup.as_ref().ok().map(|(op, _, _, _)| op.epoch);
        future_to_promise(async move {
            let result=async{
    let (op,selected,service,characteristic)=setup?;
    let device=resolve(selected).await?;op.check()?;
    state.borrow_mut().device=Some(device.clone());
    let gatt=get(&device,"gatt")?;
    let connected=invoke(&gatt,"connect",&[]).await;
    if let Err(e) = op.check() { let reused = gatt_reused(&gatt); if !reused { let _ = call(&gatt, "disconnect", &[]); } return Err(e); }
    connected?;
    let svc=invoke(&gatt,"getPrimaryService",&[service.clone().into()]).await?;op.check()?;
    let c=invoke(&svc,"getCharacteristic",&[characteristic.clone().into()]).await?;op.check()?;
    let weak=Rc::downgrade(&state);
    let listener=Closure::<dyn FnMut(JsValue)>::new(move |event:JsValue|{
     if let Some(state)=weak.upgrade(){let mut s=state.borrow_mut();if !s.subscribed{return;}
      match get(&event,"target").and_then(|t|get(&t,"value")).and_then(|v|packet(&v)){
       Ok(p)=>s.events.push(p),Err(_)=>s.events.errors=s.events.errors.saturating_add(1)
      }
     }
    });
    let weak=Rc::downgrade(&state);
    let disconnected=Closure::<dyn FnMut(JsValue)>::new(move |_|{if let Some(state)=weak.upgrade(){let mut s=state.borrow_mut();s.epoch=s.epoch.wrapping_add(1);s.busy=false;s.subscribed=false;}});
    // Store listeners before registering, so any registration failure can clean both up.
    {let mut s=state.borrow_mut();s.characteristic=Some(c.clone());s.listener=Some(listener);s.disconnected=Some(disconnected);}
    {let s=state.borrow();call(&c,"addEventListener",&["characteristicvaluechanged".into(),s.listener.as_ref().unwrap().as_ref().clone()])?;
    call(&device,"addEventListener",&["gattserverdisconnected".into(),s.disconnected.as_ref().unwrap().as_ref().clone()])?;}
    Ok(json!({"ok":true,"device":{"id":text(&device,"id"),"name":text(&device,"name"),"id_kind":"browser-scoped; not a MAC address"},"service_uuid":service,"characteristic_uuid":characteristic,"properties":properties(&c)}))
   }.await;
            // Cancellation must not tear down a newer connection.
            if result.is_err() && Some(state.borrow().epoch) == epoch {
                state.borrow_mut().cleanup();
            }
            Ok(response(&state, result).into())
        })
    }
    pub async fn read(&self) -> String {
        let result=async{let op=Operation::begin(&self.state)?;let c=op.characteristic()?;
   let v=invoke(&c,"readValue",&[]).await?;op.check()?;let p=packet(&v)?;
   Ok(json!({"ok":true,"bytes":p.len,"hex":hex(&p.bytes[..p.len]),"timestamp_ms":p.timestamp_ms}))}.await;
        response(&self.state, result)
    }
    pub async fn write(&self, hex_value: String) -> String {
        let result = async {
            let mut bytes = [0; MAX_VALUE];
            let len = decode(&hex_value, &mut bytes)?;
            let op = Operation::begin(&self.state)?;
            let c = op.characteristic()?;
            let p = properties(&c);
            let method = if p["write"] == true {
                "writeValueWithResponse"
            } else if p["write_without_response"] == true {
                "writeValueWithoutResponse"
            } else {
                return Err("Characteristic does not support writing".into());
            };
            invoke(&c, method, &[Uint8Array::from(&bytes[..len]).into()]).await?;
            op.check()?;
            Ok(json!({"ok":true,"bytes":len,"write_method":method}))
        }
        .await;
        response(&self.state, result)
    }
    pub async fn start_notifications(&self) -> String {
        let result = async {
            let op = Operation::begin(&self.state)?;
            let c = op.characteristic()?;
            self.state.borrow_mut().subscribed = true;
            let result = invoke(&c, "startNotifications", &[]).await;
            op.check()?;
            if let Err(e) = result {
                self.state.borrow_mut().subscribed = false;
                return Err(e);
            }
            Ok(json!({"ok":true}))
        }
        .await;
        response(&self.state, result)
    }
    pub async fn stop_notifications(&self) -> String {
        let result = async {
            let op = Operation::begin(&self.state)?;
            let c = op.characteristic()?;
            invoke(&c, "stopNotifications", &[]).await?;
            op.check()?;
            self.state.borrow_mut().subscribed = false;
            Ok(json!({"ok":true}))
        }
        .await;
        response(&self.state, result)
    }
    pub fn drain_notifications(&self) -> String {
        let value = {
            let mut s = self.state.borrow_mut();
            let notifications:Vec<Value>=s.events.items[..s.events.len].iter().map(|p|json!({"bytes":p.len,"hex":hex(&p.bytes[..p.len]),"timestamp_ms":p.timestamp_ms})).collect();
            let v = json!({"ok":true,"notifications":notifications,"dropped":s.events.dropped,"errors":s.events.errors});
            s.events = Events::default();
            v
        };
        response(&self.state, Ok(value))
    }
    pub fn disconnect(&self) -> String {
        {
            let mut s = self.state.borrow_mut();
            s.epoch = s.epoch.wrapping_add(1);
            s.cleanup();
        }
        response(&self.state, Ok(json!({"ok":true})))
    }
}
impl Default for BrowserBluetooth {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for BrowserBluetooth {
    fn drop(&mut self) {
        let mut s = self.state.borrow_mut();
        s.epoch = s.epoch.wrapping_add(1);
        s.cleanup();
    }
}
