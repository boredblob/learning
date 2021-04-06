import './App.css';
import { TextField, Typography } from '@material-ui/core'
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const target_max_length = 10;
const input_max_length = 17;

function App() {
  const [text, setText] = useState("");
  const [encoder, setEncoder] = useState(null);
  const [decoder, setDecoder] = useState(null);
  const [vocab, setVocab] = useState(null);
  useEffect(() => {
    async function loadModel() {
      setEncoder(await tf.loadLayersModel("./models/encoder/model.json"));
      setDecoder(await tf.loadLayersModel("./models/decoder/model.json"));
      setVocab(await loadVocab("./english.json", "./french.json"));
    }
    loadModel();
  }, []);
  return (
    <div>
      <Typography variant="h3" color="textPrimary">French:</Typography>
      <TextField id="french" variant="standard" label="" defaultValue={""} onChange={e => {
        setText(preprocess(e.target.value));
      }}/>
      <Typography variant="h3" color="textPrimary">English:</Typography>
      <TextField id="english" variant="standard" label="" value={translate(text, encoder, decoder, vocab)}/>
    </div>
  );
}

function translate(text, encoder, decoder, vocab) {
  if (encoder == null || decoder == null || vocab == null) return "";
  if (text === "") return text;
  let inputs = [];
  for (let i of text.split(' ')) {
    inputs.push(vocab.input[i]);
  }
  console.log(inputs);
  for (let i = inputs.length; i < input_max_length; i++) inputs.push(0);
  let inputTensor = tf.tensor([inputs]);
  let result = "";
  let hidden = tf.zeros([1, 1024]);
  let [, enc_hidden] = encoder.predict([inputTensor, hidden]);
  let dec_hidden = enc_hidden;
  let dec_input = tf.expandDims([vocab.target['<start>']], 0);

  for (let i = 0; i < target_max_length; i++) {
    let predictions;
    [predictions, dec_hidden] = decoder.predict([dec_input, dec_hidden]);
    let predicted_id = argMax(predictions.gather(0).gather(0).arraySync());
    if (Object.keys(vocab.target)[predicted_id] === "<end>") break;
    if (Object.keys(vocab.target)[predicted_id] === "<start>") break;
    console.log(predicted_id);
    result += Object.keys(vocab.target)[predicted_id] + ' ';
    dec_input = tf.expandDims([predicted_id], 0);
  }
  return result;
}

function preprocess(str) {
  let w = str.toLowerCase().trim();
  w = w.replace(/([?.!,¿¡])/, " " + w[w.search(/([?.!,¿¡])/)] + " ");
  w = w.replace(/[" "]+/, " ");
  w = w.replace(/[^a-zA-Z?.!,¿¡]+/, " ");
  w = w.trim();
  w = "<start> " + w + " <end>";
  return w;
}

async function loadVocab(english_path, french_path) {
  // const lines = (await fetch(path).then(r => r.text())).trim().split("\n").slice(0, 1000);
  // let input_lang = [];
  // let target_lang = [];
  // for (let line of lines) {
  //   input_lang.push(preprocess(line.split("\t")[0]));
  //   target_lang.push(preprocess(line.split("\t")[1]));
  // }
  // console.log(input_lang, target_lang);

  const englishDict = await fetch(english_path).then(r => r.json());
  const frenchDict = await fetch(french_path).then(r => r.json());
  return {input: frenchDict, target: englishDict};
}

function argMax(arr) {
  if (arr.length === 0) return -1;
  let max = arr[0];
  let maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }

  return maxIndex;
}

export default App;
