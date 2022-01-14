from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray

directory = "D:\\Dataset\\celebA\\img_align_celeba\\img_align_celeba\\"

def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels

def extract_face(model, pixels, required_size=(80,80)):
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None
    
    x1, y1, width, height = faces[0]['box']
    x1 = abs(x1)
    y1 = abs(y1)
    x2 = x1+width
    y2 = y1+height
    
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_faces(filenames):
#     print(filenames)
    model = MTCNN()
    faces = []
    for filename in filenames:
        pixels = load_image(directory+filename)
        face = extract_face(model, pixels)
        if face is None:
            continue
        faces.append(face)
#         print(len(faces), face.shape)
    return asarray(faces)