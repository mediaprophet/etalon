/* Original byte-buffer skeleton using the public QAIC forward-loader ABI. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "remote.h"

#ifndef ETALON_PROVIDER_ID
#error "Define ETALON_PROVIDER_ID for this DSP image"
#endif
#ifndef ETALON_MAX_PAIRS
#error "Define ETALON_MAX_PAIRS for this DSP image"
#endif
#if ETALON_PROVIDER_ID == 0
#error "Provider identity must be nonzero"
#endif
#if ETALON_MAX_PAIRS < 1 || ETALON_MAX_PAIRS > 1048576
#error "Invalid IQ power image capacity"
#endif

extern int32_t etalon_iq_power_open(uint64_t, uint32_t, const uint8_t *, size_t, uint64_t *);
extern int32_t etalon_iq_power_close(uint64_t);
extern int32_t etalon_iq_power_process(uint64_t, uint32_t, uint64_t,
    const uint8_t *, size_t, uint8_t *, size_t);

static int open_context(remote_arg *args) {
    int32_t uri_len;
    uint64_t handle = 0;
    int32_t status;
    if (args[0].buf.pv == NULL || args[0].buf.nLen != sizeof(uri_len)
        || args[1].buf.pv == NULL || args[1].buf.nLen < 2
        || args[1].buf.nLen > 513) return -1;
    /* The primitive length need not be naturally aligned. */
    memcpy(&uri_len, args[0].buf.pv, sizeof(uri_len));
    if (uri_len < 2 || (size_t)uri_len != args[1].buf.nLen) return -1;
    status = etalon_iq_power_open((uint64_t)ETALON_PROVIDER_ID,
        (uint32_t)ETALON_MAX_PAIRS, (const uint8_t *)args[1].buf.pv,
        (size_t)uri_len, &handle);
    if (status == 0) args[2].h64 = (remote_handle64)handle;
    return (int)status;
}

#if defined(__GNUC__)
__attribute__((visibility("default")))
#endif
int etalon_iq_power_skel_handle_invoke(remote_handle64 handle,
    uint32_t scalars, remote_arg *args) {
    /* SDK owns a live argument array of the scalar-declared length. */
    if (args == NULL) return -1;
    if (scalars == REMOTE_SCALARS_MAKEX(0, 0, 2, 0, 0, 1)) {
        return open_context(args);
    }
    if (scalars == REMOTE_SCALARS_MAKEX(0, 1, 0, 0, 1, 0)) {
        return (int)etalon_iq_power_close((uint64_t)args[0].h64);
    }
    if (scalars != REMOTE_SCALARS_MAKE(2, 1, 1)) return -1;
    if (args[0].buf.pv == NULL || args[1].buf.pv == NULL
        || args[0].buf.nLen < 64 || args[1].buf.nLen < 64
        || args[0].buf.nLen > 64u + (size_t)ETALON_MAX_PAIRS * 4u
        || args[1].buf.nLen > 64u + (size_t)ETALON_MAX_PAIRS * 8u) return -1;
    return (int)etalon_iq_power_process((uint64_t)ETALON_PROVIDER_ID,
        (uint32_t)ETALON_MAX_PAIRS, (uint64_t)handle,
        (const uint8_t *)args[0].buf.pv, args[0].buf.nLen,
        (uint8_t *)args[1].buf.pv, args[1].buf.nLen);
}
