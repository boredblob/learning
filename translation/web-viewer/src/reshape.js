import * as tf from '@tensorflow/tfjs';

class CustomReshape extends tf.layers.Layer {
  constructor() {
    super({});
    this.supportsMasking = true;
  }

  computeOutputShape(inputShape) {
    return [-1, inputShape[2]]
  }

  call(inputs, kwargs) {
    let input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }
    this.invokeCallHook(inputs, kwargs);
    return tf.reshape(input, [-1, input.shape[2]]);
  }

  static get className() {
    return "CustomReshape";
  }
}

tf.serialization.registerClass(CustomReshape);  // Needed for serialization.

export function customreshape() {
  return new CustomReshape();
}