#include "../common.cuh"
#include "gfx906-fattn-q8.cuh"

void ggml_cuda_flash_attn_ext_tile_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        // Phase 5: Cases 40, 80, 112 temporarily disabled - not multiples of 32 (Q8_0 block size)
        // TODO: Re-enable after adding support for non-32-multiple head sizes
        // case  40: {
        //     GGML_ASSERT(V->ne[0] == K->ne[0]);
        //     ggml_cuda_flash_attn_ext_tile_q8_case< 40,  40>(ctx, dst);
        // } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case< 64,  64>(ctx, dst);
        } break;
        // case  80: {
        //     GGML_ASSERT(V->ne[0] == K->ne[0]);
        //     ggml_cuda_flash_attn_ext_tile_q8_case< 80,  80>(ctx, dst);
        // } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case< 96,  96>(ctx, dst);
        } break;
        // case 112: {
        //     GGML_ASSERT(V->ne[0] == K->ne[0]);
        //     ggml_cuda_flash_attn_ext_tile_q8_case<112, 112>(ctx, dst);
        // } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case<128, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case<256, 256>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_q8_case<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size for Q8 tile kernel (Phase 5: requires DKQ multiple of 32)");
        } break;
    }
}
