from PIL import Image, ImageOps


image_path = "test_hat_image_1.jpg"
def resize_with_padding(image_path, output_path, target_size=(1080, 720)):
    image = Image.open(image_path).convert("RGB")

    # Resize while keeping aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)

    # Create a new image with the target size and white background
    new_image = Image.new("RGB", target_size, (255, 255, 255))  # White background

    # Paste the resized image in the center
    x_offset = (target_size[0] - image.size[0]) // 2
    y_offset = (target_size[1] - image.size[1]) // 2
    new_image.paste(image, (x_offset, y_offset))

    new_image.save(output_path)
    print(f"Saved resized image: {output_path}")

# Example usage
resize_with_padding(image_path, "resized_test_image_2.jpg")
