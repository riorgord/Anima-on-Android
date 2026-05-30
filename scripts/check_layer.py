import os, ctypes
# Set env var BEFORE loading the .so — the Vulkan loader reads it at vkCreateInstance
os.environ['VK_LAYER_PATH'] = '/data/local/tmp'
print("VK_LAYER_PATH:", os.environ.get('VK_LAYER_PATH', 'NOT SET'))

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print("init:", ok)
lib.dit_destroy()
