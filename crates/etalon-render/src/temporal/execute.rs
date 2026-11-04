use super::*;

impl<R> TemporalResolveGpu<R> {
    /// Current render extent.
    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// `true` when the history buffer holds a valid previous frame.
    pub fn history_valid(&self) -> bool {
        self.history_valid
    }

    /// `true` when the resolved output is ready.
    pub fn resolved_ready(&self) -> bool {
        self.resolved_ready
    }

    /// Current submission state.
    pub fn submission_state(&self) -> TemporalSubmissionState {
        self.submission_state
    }

    /// The renderer-owned current-frame linear-depth producer target.
    pub fn linear_depth_view(&self) -> &etalon::TextureView {
        &self.linear_depth_view
    }

    /// Convert the authoritative depth attachment into view-space linear depth.
    ///
    /// This pass is recorded after opaque/transparent scene rendering and before
    /// temporal resolve. It never treats the depth attachment itself as a colour
    /// view, so the producer contract remains valid on portable WebGPU backends.
    pub fn record_linear_depth_producer(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        encoder: &mut etalon::CommandEncoder,
        scene_depth: &etalon::TextureView,
        near_plane: f32,
        far_plane: f32,
    ) -> bool {
        if !near_plane.is_finite()
            || !far_plane.is_finite()
            || near_plane <= 0.0
            || far_plane <= near_plane
        {
            return false;
        }
        let params = DepthProducerParams {
            near_plane,
            far_plane,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
        };
        queue.write_buffer(
            &self.depth_producer_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("qualia-temporal-depth-producer-bind"),
            layout: &self.depth_producer_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureView(scene_depth),
                },
                buffer_entry(1, &self.depth_producer_params_buf),
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("qualia-temporal-depth-producer"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &self.linear_depth_view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.depth_producer_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        self.submission_state = TemporalSubmissionState::LinearDepthReady;
        true
    }

    /// Discard history after a resize, camera cut, seek, or device recovery.
    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
        self.resolved_ready = false;
        self.submission_state = TemporalSubmissionState::Idle;
    }

    /// The scene-linear resolve target for an output-chain handoff.
    pub fn resolved_view(&self) -> &etalon::TextureView {
        &self.resolved_view
    }

    /// Record one resolve and, when requested, publish colour and linear depth history.
    ///
    /// Every producer view is mandatory. The method returns `false` for an extent
    /// mismatch and records no commands, which is the fail-closed path for a
    /// caller that has not yet synchronized its attachments.
    pub fn record_resolve(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        encoder: &mut etalon::CommandEncoder,
        inputs: TemporalResolveInputs<'_, '_>,
        reset_history: bool,
        publish_history: bool,
    ) -> bool {
        let dimensions = [self.width as f32, self.height as f32];
        if inputs.extent != (self.width, self.height)
            || dimensions
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return false;
        }

        let history_bit = if self.history_valid && !reset_history {
            1.0
        } else {
            0.0
        };
        let reset_bit = if reset_history { 2.0 } else { 0.0 };
        let params = TemporalParams {
            viewport: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            controls: [
                DEPTH_DISOCCLUSION_THRESHOLD,
                MAX_MOTION_FOR_HISTORY,
                MAX_HISTORY_WEIGHT,
                history_bit + reset_bit,
            ],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("qualia-temporal-resolve-bind"),
            layout: &self.bind_layout,
            entries: &[
                texture_entry(0, inputs.current_scene),
                texture_entry(1, &self.history_view),
                texture_entry(2, inputs.current_linear_depth),
                texture_entry(3, &self.history_depth_view),
                texture_entry(4, inputs.motion_vectors),
                texture_entry(5, inputs.reactive_mask),
                buffer_entry(6, &self.params_buf),
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: Some("qualia-temporal-resolve"),
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &self.resolved_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.submission_state = TemporalSubmissionState::Resolved;

        if publish_history {
            let depth_bind = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("qualia-temporal-depth-publication-bind"),
                layout: &self.publish_depth_layout,
                entries: &[texture_entry(7, inputs.current_linear_depth)],
            });
            let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: Some("qualia-temporal-depth-publication"),
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &self.history_depth_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.publish_depth_pipeline);
            pass.set_bind_group(0, &depth_bind, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);

            encoder.copy_texture_to_texture(
                etalon::TexelCopyTextureInfo {
                    texture: &self.resolved_texture,
                    mip_level: 0,
                    origin: etalon::Origin3d::ZERO,
                    aspect: etalon::TextureAspect::All,
                },
                etalon::TexelCopyTextureInfo {
                    texture: &self.history_texture,
                    mip_level: 0,
                    origin: etalon::Origin3d::ZERO,
                    aspect: etalon::TextureAspect::All,
                },
                etalon::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.history_valid = true;
            self.submission_state = TemporalSubmissionState::HistoryPublished;
        } else if reset_history {
            self.history_valid = false;
        }
        self.resolved_ready = true;
        true
    }

    /// Record the complete renderer-owned temporal handoff in schedule order.
    ///
    /// This is the explicit host submission seam: the host supplies all producer views and an
    /// output target, while this owner records resolve, history publication, and final output in
    /// that order. A disabled or incomplete schedule fails closed before any command is recorded.
    pub fn record_scheduled_output(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        encoder: &mut etalon::CommandEncoder,
        inputs: TemporalResolveInputs<'_, '_>,
        target: &etalon::TextureView,
        schedule: TemporalOutputSchedule,
        exposure_ev: f32,
        white_balance_gains: [f32; 3],
        surface_is_srgb: bool,
    ) -> bool {
        if !self.scheduled_output_is_admissible(
            inputs.extent,
            schedule,
            exposure_ev,
            white_balance_gains,
        ) {
            return false;
        }
        // The graph's lifecycle state is authoritative for this submission. This prevents a
        // stale owner-side history from being read when the scheduler deliberately starts a
        // frame without previous-frame history but does not need a destructive reset request.
        if !schedule.reads_history {
            self.history_valid = false;
        }
        if !self.record_resolve(
            device,
            queue,
            encoder,
            inputs,
            schedule.reset_history,
            schedule.publish_history,
        ) {
            return false;
        }
        let output_recorded = self.record_final_output(
            device,
            queue,
            encoder,
            target,
            exposure_ev,
            white_balance_gains,
            surface_is_srgb,
        );
        if !output_recorded {
            self.invalidate_history();
        }
        output_recorded
    }

    /// Preflight a scheduled submission before an adapter records any renderer-owned producer
    /// passes. The host extent is the only attachment metadata available at this API boundary;
    /// WebGPU validates the actual texture-view dimensions when the command buffer is submitted.
    pub fn scheduled_output_is_admissible(
        &self,
        extent: (u32, u32),
        schedule: TemporalOutputSchedule,
        exposure_ev: f32,
        white_balance_gains: [f32; 3],
    ) -> bool {
        schedule.is_valid()
            && extent == (self.width, self.height)
            && !(schedule.reads_history && !self.history_valid)
            && exposure_ev.is_finite()
            && white_balance_gains
                .iter()
                .all(|gain| gain.is_finite() && *gain >= 0.0)
            && (self.exposure_scale)(exposure_ev).is_some()
    }

    /// Record the existing versioned SDR transform against the resolved scene.
    ///
    /// This is an explicit handoff, not an automatic second output pass. Hosts
    /// must choose either this handoff or the existing `OutputChain` composite.
    pub fn record_final_output(
        &mut self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        encoder: &mut etalon::CommandEncoder,
        target: &etalon::TextureView,
        exposure_ev: f32,
        white_balance_gains: [f32; 3],
        surface_is_srgb: bool,
    ) -> bool {
        if !self.resolved_ready
            || !matches!(
                self.submission_state,
                TemporalSubmissionState::Resolved | TemporalSubmissionState::HistoryPublished
            )
            || !exposure_ev.is_finite()
            || white_balance_gains
                .iter()
                .any(|gain| !gain.is_finite() || *gain < 0.0)
        {
            return false;
        }
        let Some(exposure) = (self.exposure_scale)(exposure_ev) else {
            return false;
        };
        let params = OutputParams {
            exposure,
            surface_is_srgb: if surface_is_srgb { 1.0 } else { 0.0 },
            _padding0: [0.0; 2],
            white_balance_gains,
            _padding1: 0.0,
        };
        queue.write_buffer(&self.output_params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("qualia-temporal-final-output-bind"),
            layout: &self.output_layout,
            entries: &[
                texture_entry(0, &self.resolved_view),
                buffer_entry(1, &self.output_params_buf),
            ],
        });
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("qualia-temporal-final-output"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.output_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        self.submission_state = TemporalSubmissionState::OutputReady;
        true
    }
}
