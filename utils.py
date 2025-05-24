from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import cv2

animal_names = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly",
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow",
    "deer", "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant",
    "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper",
    "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird",
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard",
    "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter",
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig",
    "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros",
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid",
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf",
    "wombat", "woodpecker", "zebra"
]

merge_map = {
    "deer": "Deer/Antelope",
    "antelope": "Deer/Antelope",
    "bison": "Bison/Ox/Cow",
    "ox": "Bison/Ox/Cow",
    "cow": "Bison/Ox/Cow",
    "bear": "Bear/Boar",
    "boar": "Bear/Boar",
    "fly": "Fly/Mosquito",
    "mosquito": "Fly/Mosquito",
    "butterfly": "Butterfly/Moth",
    "moth": "Butterfly/Moth",
    "beetle": "Beetle/Ladybug",
    "ladybugs": "Beetle/Ladybug",
    "crab": "Crustaceans",
    "lobster": "Crustaceans",
    "dolphin": "Dolphin/Whale",
    "whale": "Dolphin/Whale",
    "raccoon": "Raccoon/Possum",
    "possum": "Raccoon/Possum",
    "hamster": "Rodents",
    "mouse": "Rodents",
    "rat": "Rodents",
    "duck": "Waterfowl",
    "goose": "Waterfowl",
    "swan": "Waterfowl",
    "horse": "Horse/Donkey",
    "donkey": "Horse/Donkey",
    "pigeon": "Small Birds",
    "sparrow": "Small Birds",
    "otter": "Otter/Seal",
    "seal": "Otter/Seal",
    "chimpanzee": "Great Apes",
    "gorilla": "Great Apes",
    "orangutan": "Great Apes",
    "octopus": "Octopus/Squid",
    "squid": "Octopus/Squid",
    "flamingo": "Flamingo/Pelecaniformes",
    "pelecaniformes": "Flamingo/Pelecaniformes",
    "rhinoceros": "Elephant/Rhino",
    "elephant": "Elephant/Rhino"
}


def load_images(folder_path, img_shape=(224, 224)):
    data_path = os.path.join(folder_path, "data/animals")

    X = []
    y = []

    i = 1

    for animal_path in os.listdir(data_path):
        for img_path in os.listdir(os.path.join(data_path, animal_path)):
            print(f"Image: {i}")
            i += 1

            full_path = os.path.join(data_path, animal_path, img_path)
            img = cv2.imread(full_path)
            img = cv2.resize(img, img_shape)

            X.append(img)

            label = animal_names.index(animal_path)
            y.append(label)

    np.save(f"data/images_{img_shape[0]}.npy", X)
    np.save("data/labels.npy", y)

    print("Successfully loaded and images")


def get_merged_animal_names():
    merged_animal_names = []
    for animal in animal_names:
        if animal in merge_map:
            merged_label = merge_map[animal]
            if merged_label not in merged_animal_names:
                merged_animal_names.append(merged_label)
        else:
            if animal not in merged_animal_names:
                merged_animal_names.append(animal)

    return merged_animal_names


def get_merged_animal_labels(labels):
    merged_labels = []
    merged_animal_names = get_merged_animal_names()

    for label in labels:
      animal_name = animal_names[label]
      if animal_name in merge_map.keys():
        new_group = merge_map[animal_name]
        new_label = merged_animal_names.index(new_group)
      else:
        new_label = merged_animal_names.index(animal_name)
      merged_labels.append(new_label)

    merged_labels = np.array(merged_labels, dtype=np.int16)
    return merged_labels


def construct_confusion_matrix(animal_labels, y_true, y_pred, num_top=10):
    cm = confusion_matrix(y_true, y_pred)

    np.fill_diagonal(cm, 0)

    flat = cm.flatten()
    top_indices = flat.argsort()[::-1][:num_top]

    for idx in top_indices:
        true_label = idx // cm.shape[1]
        pred_label = idx % cm.shape[1]
        confusion_count = cm[true_label, pred_label]
        if confusion_count > 0:
            print(f"True: {animal_labels[true_label]:<15} Predicted: {animal_labels[pred_label]:<15} Confused {confusion_count} times")


def plot_history(history):
    # Accuracy
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.ylim(0)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
