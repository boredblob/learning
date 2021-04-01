import './App.css';
import { Button, Slider } from '@material-ui/core';
import * as tf from '@tensorflow/tfjs';
import { useEffect, useReducer, useState } from "react";

const number_of_switches = 100;

function App() {
  const [seed, updateSeed] = useReducer(seedReducer, getRandomValues(), getRandomValues);
  return (
    <main className="container">
      <div className="modelView-container">
        <ModelView key={JSON.stringify(seed)} seed={seed} width={400} height={400}></ModelView>
      </div>
      <div className="button-container">
        <Button variant="contained" color="primary" onClick={() => {updateSeed({type: "reset"})}}>Set Random</Button>
        <Button variant="contained" color="primary" onClick={() => {updateSeed({type: "zero"})}}>Set Zero</Button>
      </div>
      <div className="sliders-container">
        <Sliders key={JSON.stringify(seed)} seed={seed} updateSeed={updateSeed}></Sliders>
      </div>
    </main>
  );
}

function ModelView(props) {
  const [model, setModel] = useState(null);
  const [imageURL, setImageURL] = useState("");
  useEffect(() => {
    async function loadModel() {
      setModel(await tf.loadLayersModel("./model/model.json"));
    }
    loadModel();
  }, []);
  if (model == null) {
    return (
      <div className="placeholder" style={{width: props.width + "px", height: props.height + "px"}}></div>
    )
  } else {
    const image = model.predict(tf.tensor(props.seed, [1, 100]));
    const imageCanvas = document.createElement("canvas");
    imageCanvas.width = props.width;
    imageCanvas.height = props.height;
    tf.browser.toPixels(image.reshape([28, 28, 1]).div(2).add(0.5), imageCanvas).then(() => {
      setImageURL(imageCanvas.toDataURL());
      image.dispose();
    });
    return (
      <img className="modelView" src={imageURL} alt="" width={props.width} height={props.height}></img>
    );
  }
}

function getRandomValues() {
  let arr = [];
  for (let i = 0; i < number_of_switches; i++) {
    arr.push(Math.random())
  }
  return arr;
}

function Sliders(props) {
  let arr = [];
  for (let i = 0; i < props.seed.length; i++) {
    arr.push((
      <Slider className="slider" color="primary" key={i} orientation="vertical" min={0} max={1} step={0.00000001} defaultValue={props.seed[i]} onChangeCommitted={(e, val) => {
        props.updateSeed({type: "update", payload: {key: i, value: val}});
      }}></Slider>
    ))
  }
  return arr;
}

function seedReducer(seed, action) {
  switch (action.type) {
    case "update":
      seed[action.payload.key] = action.payload.value;
      return Array.from(seed);
    case "reset":
      return getRandomValues();
    case "zero":
      return new Array(100).fill(0);
    default:
      throw new Error();
  }
}

export default App;
