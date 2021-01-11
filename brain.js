const brain = require('brain.js');

const normalize = (val, min, max) => (val - min) / (max - min);
// const normalize = (val, min, max) => Math.log(val);

const expand = (val, min, max) => val * (max - min) + min;
// const expand = (val, min, max) => Math.exp(val);


// const min = 4;
// const max = 230561;
// const pattern = [
//     0,
//     normalize(5, min, max),
//     normalize(11, min, max),
//     normalize(34, min, max),
//     normalize(137, min, max),
//     normalize(686, min, max),
//     normalize(4117, min, max),
//     normalize(28820, min, max),
//     1
// ];
// const next = 2075050;

const min = 4;
const max = 4194304;
const patternStart = [
    0.001,
    normalize(8, min, max),
    normalize(16, min, max),
    normalize(32, min, max),
    normalize(64, min, max),
    normalize(128, min, max),
    normalize(256, min, max),
    normalize(512, min, max),
    normalize(1024, min, max),
    normalize(2048, min, max),
    normalize(4096, min, max),
    normalize(8192, min, max),
    normalize(16384, min, max),
];
const patternEnd = [
    normalize(32768, min, max),
    normalize(65536, min, max),
    normalize(131072, min, max),
    normalize(262144, min, max),
    normalize(524288, min, max),
    normalize(1048576, min, max),
    normalize(2097152, min, max),
    1
];
const next = 8388608;

const pattern = [
    patternStart,
    patternEnd
];

const net = new brain.recurrent.LSTMTimeStep({
    hiddenLayers: [10],
    activation: 'tanh'
});

console.time('Training');

let error = 1;
const ERROR_THRESH = 0.0075;
while (error > ERROR_THRESH) {

    // net.train([pattern], {
    //     iterations: 1000,
    //     errorThresh: ERROR_THRESH,
    //     learningRate: 0.0005,
    //     log: true,
    // });

    error = net.train(pattern, {
        iterations: 500,
        errorThresh: ERROR_THRESH,
        learningRate: 0.005,
        // log: true,
    }).error;

    // error = net.train([pattern], {
    //     iterations: 1,
    //     errorThresh: ERROR_THRESH,
    //     learningRate: 0.05
    //     // log: true,
    // }).error;

    // error = net.train([pattern], {
    //     iterations: 1,
    //     errorThresh: ERROR_THRESH,
    //     learningRate: 0.01,
    //     log: true
    // }).error;

    // error = net.train([pattern], {
    //     iterations: 2000,
    //     errorThresh: ERROR_THRESH,
    //     learningRate: 0.0015,
    //     log: true
    // }).error;

    // error = net.train([pattern], {
    //     iterations: 1000,
    //     errorThresh: ERROR_THRESH,
    //     learningRate: 0.006,
    //     log: true
    // }).error;

}
console.timeEnd('Training');

const prediction = expand(net.run(patternEnd), min, max);
console.log(prediction);
console.log(next);
console.log(next - prediction)
console.log((next - prediction) / next * 100);