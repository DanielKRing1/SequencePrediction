// TODO
// 1. Push this to Github
// 2. Push brain.js version to Github
// 3. Compare and adjust this version
// 4. Make music
// 5. Make Lemonade: Rate Trend: Fetch TrendTracker from TrandCache -> Call rate on TrendTracker -> Calls rate on all RelQuerents, EntityQuerent, and DayQuerent
// 6. Store these data in Redux
// 7. Call all updates to static Caches from Redux Thunks


const tf = require('@tensorflow/tfjs');

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
require('@tensorflow/tfjs-node');

// // Train a simple model:
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
// model.add(tf.layers.dense({units: 1, activation: 'linear'}));
// model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
console.log(xs)
// const ys = tf.randomNormal([100, 1]);
// console.log(ys)

// model.fit(xs, ys, {
//   epochs: 100,
//   callbacks: {
//     onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
//   }
// });
  



function normalizeSequence(sequence) {
    const min = sequence[0];
    const max = sequence[sequence.length - 1];

    return [
        sequence.map((val) => normalize(val, min, max)),
        min,
        max
    ];
}
function normalize(val, min, max) {
    return (val - min) / (max - min);
}

function denormalize(val, min, max) {
    return val * (max - min) + min;
}

function denormalizeTensor(tensor, min, max) {
    return tensor.mul(max - min).add(min);
}


const SEQUENCE_LEN = 4;

const SEQUENCE = [1, 2, 6, 21, 88, 445, 2676, 18739, 149920];
const [ pattern, MIN, MAX ] = normalizeSequence(SEQUENCE);

const inputs = [];
const outputs = [];

for(let i = 0; i < pattern.length - SEQUENCE_LEN; i++) {
    const start = i;
    const stop = start + SEQUENCE_LEN;
    
    const input = pattern.slice(start, stop).map(val => [val]);
    const output = [pattern[stop]];

    inputs.push(input);
    outputs.push(output);
}

console.log('Raw inputs and outputs');
console.log(inputs);
console.log(outputs);

// Elu
// 10 units
// 3046, 3074
// 20 units
// 3088, 2975, 4468
// 100 units
// 3180

// Relu
// BAD

// Linear
// 10 units
// 3025
// 20 units
// 2997, 2998, 3015
// 100 units
// 3080, 3734,3120

// Tanh
// 10 units
// 3103
// 20 units
// 2718, 1908, 2799
// 100 units
// 2799, 1890, 2817

function createSequencePredictionModel() {
    const model = tf.sequential();

    model.add(tf.layers.lstm({
        units: 20,
        inputShape: [SEQUENCE_LEN, 1]
    }));
    // model.add(tf.layers.flatten({
    //     inputShape: [SEQUENCE_LEN, 1]
    // }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'tanh'
    }));
    // model.add(tf.layers.dense({
    //     units: 1,
    //     activation: 'sigmoid'
    // }));


    const learningRate = 4e-3;
    const loss = 'meanSquaredError';
    const optimizer = tf.train.adam(learningRate);

    model.compile({
        loss,
        optimizer,
        metrics: ['accuracy']
    });

    return model;
}

async function trainModel(model) {
    console.log('Tensor inputs and outputs');
    tf.tensor(inputs).print();
    tf.tensor(outputs).print();

    const smallLearningRate = 0.0005;
    const bigLearningRate = 0.005
    const biggerLearningRate = 0.05;
    const escapeLearningRate = 0.05;


    model.myFlag='small';
    model.myIterations = 0;

    await model.fit(tf.tensor(inputs), tf.tensor(outputs), {
        epochs: 1000,
        verbose: false,
        callbacks: {
          onEpochEnd: (epoch, log) => {
                if(epoch % 100 === 0) console.log(`Epoch ${epoch}: loss = ${log.loss}`);
                if(Math.abs(log.loss) <= 0.00001) model.stopTraining = true;

        //         model.myIterations++;
        //         switch(model.myFlag) {
        //             case 'small':
        //                 if(model.myIterations % 500 === 0) {
                            
        //                     // if(log.loss > 0.1) {
        //                     //     console.log('To escape');

        //                     //     model.optimizer.learningRate = escapeLearningRate;
        //                     //     model.myFlag = 'escape';
        //                     //     model.myIterations = 0;
        //                     // }else {
        //                         console.log('To big');

        //                         model.optimizer.learningRate = bigLearningRate;
        //                         model.myFlag = 'big';
        //                         model.myIterations = 0;
        //                     // }
        //                 }
        //                 break;

        //             case 'big':
        //                 if(model.myIterations % 10 === 0) {
        //                     console.log('To small');

        //                     model.optimizer.learningRate = smallLearningRate;
        //                     model.myFlag = 'small';
        //                     model.myIterations = 0;
        //                 }
        //                 break;

        //             case 'bigger':
        //                 if(model.myIterations % 10 === 0) {
        //                     console.log('To small');

        //                     model.optimizer.learningRate = smallLearningRate;
        //                     model.myFlag = 'small';
        //                     model.myIterations = 0;
        //                 }
        //                 break;

        //             case 'escape':
        //                 if(log.loss < 0.1) {
        //                     console.log('To small');

        //                     model.optimizer.learningRate = smallLearningRate;
        //                     model.myFlag = 'small';
        //                     model.myIterations = 0;
        //                 }
        //                 break;
        //       }
          }
        }
      });
}

function predict(model, start) {
    const stop = start + SEQUENCE_LEN;

    const test = pattern.slice(start, stop).map(val => [val]);
    console.log(test);
    const prediction = model.predict(tf.tensor([test]));
    console.log('Prediction');
    prediction.print();

    const scaledPrediction = denormalizeTensor(prediction, MIN, MAX);
    scaledPrediction.print();

    console.log(denormalize(prediction.dataSync()[0], MIN, MAX));
}

async function run() {
    console.log('Creating model\n\n');
    const model = await createSequencePredictionModel();
    
    console.log('Training model\n\n');
    await trainModel(model);

    console.log('Predicting with model\n\n');
    await predict(model, 0);
    await predict(model, 1);
    await predict(model, 2);
}

run();