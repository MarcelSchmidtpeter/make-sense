import * as tf from "@tensorflow/tfjs";

const COCO_NAMES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

// Update COCO_NAMES if your model has only 5 classes
const CUSTOM_CLASS_NAMES = ["class0", "class1", "class2", "class3", "class4"];

const INFERENCE_RESOLUTION: [number, number] = [640, 640];

export interface ModelConfig {
  source: string | File[];
  classNames?: string[];
}

export const YOLO_V5_N_COCO_MODEL_CONFIG: ModelConfig = {
  source:
    "https://raw.githubusercontent.com/SkalskiP/yolov5js-zoo/master/models/coco/yolov5n/model.json",
  classNames: COCO_NAMES,
};

export const YOLO_V5_S_COCO_MODEL_CONFIG: ModelConfig = {
  source:
    "https://raw.githubusercontent.com/SkalskiP/yolov5js-zoo/master/models/coco/yolov5s/model.json",
  classNames: COCO_NAMES,
};

export const YOLO_V5_M_COCO_MODEL_CONFIG: ModelConfig = {
  source:
    "https://raw.githubusercontent.com/SkalskiP/yolov5js-zoo/master/models/coco/yolov5m/model.json",
  classNames: COCO_NAMES,
};

export interface DetectedObject {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  classId: number;
  class?: string;
}

export class YOLOv5 {
  public model: tf.GraphModel;
  public inferenceResolution: [number, number];
  public classNames?: string[];

  constructor(
    model: tf.GraphModel,
    inferenceResolution: [number, number],
    classNames?: string[]
  ) {
    this.model = model;
    this.inferenceResolution = inferenceResolution;
    this.classNames = classNames;
  }

  public static preprocessImage(
    image: HTMLImageElement | HTMLCanvasElement,
    inferenceResolution: [number, number]
  ): [tf.Tensor4D, [number, number]] {
    // Convert the image to a tensor
    const inputTensor = tf.browser.fromPixels(image);

    // Get original image dimensions
    const [originalHeight, originalWidth] = [image.height, image.width];

    // Resize the image to the model's input resolution without maintaining aspect ratio
    const resizedTensor: tf.Tensor4D = tf.image
      .resizeBilinear(inputTensor, inferenceResolution)
      //.div(tf.scalar(255)) // Normalize to [0, 1]
      //.resizeNearestNeighbor(inputTensor, inferenceResolution)
      //.toFloat()
      .div(tf.scalar(255.0)) // Normalize to [0, 1]
      .expandDims(); // Add batch dimension

    // Dispose the original tensor to free memory
    inputTensor.dispose();

    return [resizedTensor, [originalHeight, originalWidth]];
  }

  public async detect(
    image: HTMLImageElement | HTMLCanvasElement,
    minScore: number = 0.25
  ): Promise<DetectedObject[]> {
    // Preprocess the image
    const [preprocessedTensor, inputResolution] = tf.tidy(() => {
      return YOLOv5.preprocessImage(image, this.inferenceResolution);
    });

    // Execute the model and get the output tensor
    const result = (await this.model.executeAsync(
      preprocessedTensor
    )) as tf.Tensor;

    const [batch, numAttributes, numPredictions] = result.shape; // [1, 9, 33600]
    const numClasses = numAttributes - 4; // Update based on your model's number of classes

    // Extract the data from the tensor
    const predictionsData = result.dataSync(); // Float32Array of length 1 * 9 * 33600 = 302400

    // Initialize an array to hold all detected objects
    const detections: DetectedObject[] = [];

    // Get original image dimensions
    const [originalHeight, originalWidth] = inputResolution;
    const [inferenceHeight, inferenceWidth] = this.inferenceResolution;

    // Calculate scaling factors
    const scaleX = originalWidth / inferenceWidth;
    const scaleY = originalHeight / inferenceHeight;

    // Precompute the stride for each attribute
    const stride = numPredictions;

    for (let i = 0; i < numPredictions; i++) {
      const x_center = predictionsData[i]; // Attribute 0
      const y_center = predictionsData[stride + i]; // Attribute 1
      const width = predictionsData[2 * stride + i]; // Attribute 2
      const height = predictionsData[3 * stride + i]; // Attribute 3

      // Extract class probabilities
      let maxProb = 0;
      let classId = -1;
      for (let c = 0; c < numClasses; c++) {
        const prob = predictionsData[(4 + c) * stride + i]; // Attributes 4 to 8
        if (prob > maxProb) {
          maxProb = prob;
          classId = c;
        }
      }

      // Skip if class probability is below the threshold
      if (maxProb < minScore) continue;

      // Calculate bounding box coordinates
      const minX = (x_center - width / 2) * scaleX;
      const minY = (y_center - height / 2) * scaleY;
      const maxX = (x_center + width / 2) * scaleX;
      const maxY = (y_center + height / 2) * scaleY;

      // Clamp coordinates to image boundaries
      const clampedMinX = Math.max(0, Math.min(minX, originalWidth));
      const clampedMinY = Math.max(0, Math.min(minY, originalHeight));
      const clampedMaxX = Math.max(0, Math.min(maxX, originalWidth));
      const clampedMaxY = Math.max(0, Math.min(maxY, originalHeight));

      // Create the DetectedObject
      const detectedObject: DetectedObject = {
        x: clampedMinX,
        y: clampedMinY,
        width: clampedMaxX - clampedMinX,
        height: clampedMaxY - clampedMinY,
        score: maxProb,
        classId: classId,
        class: this.classNames ? this.classNames[classId] : undefined,
      };

      detections.push(detectedObject);
    }

    // Clean up tensors
    preprocessedTensor.dispose();
    result.dispose();

    // Apply Non-Max Suppression (NMS)
    const finalDetections = await this.applyNMS(detections, 0.5, minScore);

    return finalDetections;
  }

  private async applyNMS(
    detections: DetectedObject[],
    iouThreshold: number,
    scoreThreshold: number
  ): Promise<DetectedObject[]> {
    if (detections.length === 0) return [];

    const boxes = detections.map((det) => [
      det.y,
      det.x,
      det.y + det.height,
      det.x + det.width,
    ]);
    const scores = detections.map((det) => det.score);

    const boxesTensor = tf.tensor2d(boxes);
    const scoresTensor = tf.tensor1d(scores);

    const selectedIndices = await tf.image
      .nonMaxSuppression(
        boxesTensor,
        scoresTensor,
        100,
        iouThreshold,
        scoreThreshold
      )
      .array();

    boxesTensor.dispose();
    scoresTensor.dispose();

    const finalDetections = (selectedIndices as number[]).map(
      (index) => detections[index]
    );

    return finalDetections;
  }
}

export async function load(
  config: ModelConfig,
  inputResolution: [number, number] = INFERENCE_RESOLUTION
): Promise<YOLOv5> {
  if (typeof config.source === "string") {
    return tf.loadGraphModel(config.source).then((model: tf.GraphModel) => {
      return new YOLOv5(model, inputResolution, config.classNames);
    });
  } else {
    return tf
      .loadGraphModel(tf.io.browserFiles(config.source))
      .then((model: tf.GraphModel) => {
        return new YOLOv5(model, inputResolution, config.classNames);
      });
  }
}
