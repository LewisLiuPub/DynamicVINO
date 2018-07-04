Usage examples:

	./stat_collector -d CPU -m <models>/alexnet/bvlc_alexnet_fp32.xml -i <images>/ILSVRC2012_val_00000001.bmp -t bvlc_alexnet_fp32.stats.xml

	./stat_collector -d CPU -m <models>/SSD_VGG/VGG_ILSVRC2016_SSD_300x300_deploy_fp32.xml -i <images>/ILSVRC2012_val_00000001.bmp -t VGG_ILSVRC2016_SSD_300x300_deploy_fp32.stats.xml -l ./lib/libcpu_extension.so

You can specify a folder with images in `-i` instead of one image