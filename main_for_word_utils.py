import word_utils
import helpers

if __name__ == "__main__":
    all_images,all_ground_truth = word_utils.load_lined_images("oneLine","arab_gt")
    print("__________________________")
    OCRResults_words = []
    all_OCR_results = []
    counter=0
    for images in (all_images):
        OCRResults_words = []
        for i in range(len(images)):
            word_images= word_utils.line_to_word_image(images[i])
            OCRResults_words.append(word_utils.OCRING(word_images))
        all_OCR_results.append(helpers.flatten(OCRResults_words))
        print(OCRResults_words)
        print(type(OCRResults_words))
        print(len(OCRResults_words))
        print(word_utils.word_accuracy(helpers.flatten(OCRResults_words),all_ground_truth[counter]))
        counter+=1