from common.utils import image_to_bytes

if __name__ == '__main__':
    img_bytes = image_to_bytes("./three.png")
    print(img_bytes)