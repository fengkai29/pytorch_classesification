info = {
    'loss': val_loss.data[0],
    'accuracy': val_acc.data[0]
}

for tag, value in info.items():
    logger.scalar_summary(tag, value, step)

# (2) Log values and gradients of the parameters (histogram)
for tag, value in model.named_parameters():
    tag = tag.replace('.', '/')
    logger.histo_summary(tag, to_np(value), step)
    logger.histo_summary(tag+'/grad', to_np(value.grad), step)

# (3) Log the images
info = {
    'images': to_np(img.view(-1, 28, 28)[:10])
}

for tag, images in info.items():
    logger.image_summary(tag, images, step)
