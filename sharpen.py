from PIL import Image, ImageFilter, ImageOps

# Load image
image = Image.open("850.png").convert("L")  # Convert to grayscale

# Apply sharpening
sharpened = image.filter(ImageFilter.SHARPEN)

# Enhance contrast
enhanced = ImageOps.autocontrast(sharpened)

# Apply thresholding
threshold = 190  # Adjust threshold value as needed
binary = enhanced.point(lambda p: 255 if p > threshold else 0)

# Save or display
binary.save("output.png")
binary.show()