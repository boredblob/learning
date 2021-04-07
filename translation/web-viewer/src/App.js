import './App.css';
import { createMuiTheme } from "@material-ui/core/styles";
import { TextField, ThemeProvider, Typography, Button } from '@material-ui/core';
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const input_max_length = 65;
const target_max_length = 53;
const theme = createMuiTheme({
  palette: {
    type: "dark"
  }
});

function App() {
  const [text, setText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [encoder, setEncoder] = useState(null);
  const [decoder, setDecoder] = useState(null);
  const [vocab, setVocab] = useState(null);
  const [translateError, setError] = useState(false)
  useEffect(() => {
    async function loadModel() {
      setEncoder(await tf.loadLayersModel("./models/encoder/model.json"));
      setDecoder(await tf.loadLayersModel("./models/decoder/model.json"));
      setVocab(await loadVocab());
    }
    loadModel();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <form className="container" action="" onSubmit={e => {
        e.preventDefault();
        setError(false);
        setTranslatedText(translate(text, encoder, decoder, vocab, setError));
      }}>
        <div className="textfield-wrapper">
          <Typography variant="h4" color="textPrimary">French:</Typography>
          <TextField
            className="textfield" id="french" name="french" variant="standard" label="" 
            {...(translateError && {error: true, helperText: "Unrecognised Word"})} 
            inputProps={{autoComplete: "off", spellCheck:"false"}} defaultValue={""}
            onChange={e => {
              setText(e.target.value);
            }}
          />
        </div>
        <div className="textfield-wrapper">
          <Typography variant="h4" color="textPrimary">English:</Typography>
          <TextField className="textfield" id="english" name="english" variant="standard" label="" inputProps={{autoComplete: "off", spellCheck:"false"}} value={translatedText}/>
        </div>
        <div className="button-wrapper">
          <Button id="button" type="submit" variant="contained" color="primary">Translate</Button>
        </div>
      </form>
    </ThemeProvider>
  );
}

function translate(text, encoder, decoder, vocab, setError) {
  if (encoder == null || decoder == null || vocab == null) return "";
  if (text === "") return "";
  let inputs = [];
  console.log(preprocess(text).split(" "));
  for (let i of preprocess(text).split(' ')) {
    let w = vocab.input.wi[i];
    if (w == null) {
      setError(true);
      return "";
    }
    inputs.push(w);
  }
  for (let i = inputs.length; i < input_max_length; i++) inputs.push(0);
  let inputTensor = tf.tensor([inputs]);
  let result = "";
  let hidden = tf.zeros([1, 1024]);
  let [, enc_hidden] = encoder.predict([inputTensor, hidden]);
  let dec_hidden = enc_hidden;
  let dec_input = tf.expandDims([vocab.target.wi['<start>']], 0);

  for (let i = 0; i < target_max_length; i++) {
    let predictions;
    [predictions, dec_hidden] = decoder.predict([dec_input, dec_hidden]);
    let predicted_id = argMax(predictions.gather(0).gather(0).arraySync());
    if (vocab.target.iw[predicted_id] === "<end>") break;
    result += vocab.target.iw[predicted_id] + ' ';
    dec_input = tf.expandDims([predicted_id], 0);
  }
  return result;
}

function preprocess(str) {
  let w = str.toLowerCase().trim();
  w = w.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  w = w.replace(/([?.!,¿¡])/g, " " + w[w.search(/([?.!,¿¡])/g)] + " ");
  w = w.replace(/[^a-zA-Z?.!,¿¡]+/g, " ");
  w = w.replace(/\s\s+/g, " ");
  w = w.trim();
  w = "<start> " + w + " <end>";
  return w;
}

async function loadVocab() {
  return {
    input: {
      wi: await fetch("./dicts/input_wi.json").then(r => r.json()),
      iw: await fetch("./dicts/input_iw.json").then(r => r.json())
    },
    target: {
      wi: await fetch("./dicts/target_wi.json").then(r => r.json()),
      iw: await fetch("./dicts/target_iw.json").then(r => r.json())
    }
  };
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
