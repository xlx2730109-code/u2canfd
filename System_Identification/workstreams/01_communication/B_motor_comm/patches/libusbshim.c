/* 给 aarch64 的 libdm_device.so 补 libusb_init_context：Ubuntu 22.04 自带 libusb 1.0.25，没有这个符号
   （1.0.27 起才有），这里转成 libusb_init，忽略 options。运行时用 LD_PRELOAD 加载；系统已有 libusb 1.0.27 以上时不需要。 */
#include <libusb-1.0/libusb.h>
int libusb_init_context(libusb_context **ctx, const void *options, int num_options) {
    (void)options; (void)num_options;
    return libusb_init(ctx);
}
