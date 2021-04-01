import './App.css';
import { Button, Slider } from '@material-ui/core';
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel("./model/model.json");

function App() {
  const [seed, updateSeed] = useReducer(seedReducer, getRandomValues(100), getRandomValues(100));
  return (
    <div>
      <ModelView seed={seed}></ModelView>
      <Button variant="contained"></Button>
      <div>{Sliders(seed, updateSeed)}</div>
    </div>
  );
}

function ModelView(props) {
  const image = model.predict(props.seed);
  console.log(image);
  return (
    <img alt=""></img>
  )
}

function getRandomValues(n) {
  let arr = [];
  for (let i = 0; i < n; i++) {
    arr.push(Math.random())
  }
  return arr;
}

function Sliders(seed, updateSeed) {
  let arr = [];
  for (let i = 0; i < seed.length; i++) {
    arr.push((
      <Slider key={i} orientation="vertical" defaultValue={seed[i]} min={0} max={1} onChange={e => {
        updateSeed({type: "update", payload: {key: i, value: e.target.value}})
      }}></Slider>
    ))
  }
}

function seedReducer(seed, action) {
  switch (action.type) {
    case "update":
      seed[action.payload.key] = action.payload.value;
      return seed;
    default:
      return;
  }
}

export default App;
