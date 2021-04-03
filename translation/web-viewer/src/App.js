import './App.css';
import { TextField, Typography } from '@material-ui/core'
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [text, setText] = useState("");
  const [encoder, setEncoder] = useState(null);
  const [decoder, setDecoder] = useState(null);
  useEffect(() => {
    async function loadModel() {
      setEncoder(await tf.loadGraphModel("./encoder/"));
      setDecoder(await tf.loadGraphModel("./decoder/"));
    }
    loadModel();
  }, [])
  return (
    <div>
      <Typography variant="h2" color="textPrimary">French:</Typography>
      <TextField id="french" variant="standard" label="" onChange={(_, val) => {
        setText(val);
      }}/>
      <Typography variant="h2" color="textPrimary">English:</Typography>
      <TextField id="english" variant="standard" label="" value={translate(text, encoder, decoder)}/>
    </div>
  );
}

function translate(text, encoder, decoder) {
  if (encoder == null || decoder == null) return "";
}

export default App;
