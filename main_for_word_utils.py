import word_utils
import helpers
import Benchmark
import CV_operations as cvop

if __name__ == "__main__":
    all_images,all_ground_truth,images_names, gt_names = word_utils.load_lined_images("oneLine","arab_gt")
    print("__________________________")
    OCRResults_words = []
    all_OCR_results = []
    accuracy = []
    counter=0
    all_images=all_images[:70]
    all_ground_truth=all_ground_truth[:70]
    images_names=images_names[:70]
    gt_names = gt_names[:70]
    for images in (all_images):
        OCRResults_words = []
        for i in range(len(images)):
            removed_noise = cvop.halfOp(images[i])
            word_images= word_utils.line_to_word_image(images[i],removed_noise)
            OCRResults_words.append(word_utils.OCRING(word_images))
        all_OCR_results.append(helpers.flatten(OCRResults_words))
        accuracy.append(word_utils.word_accuracy(helpers.flatten(OCRResults_words),all_ground_truth[counter]))
        print(accuracy[counter])
        counter+=1
    print(len(accuracy))
    Benchmark.create_csv("removed_noise.csv", [images_names,accuracy])