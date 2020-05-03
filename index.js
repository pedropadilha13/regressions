require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCsv = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCsv('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 20
});

regression.train();

// console.log(`Updated M: ${regression.weights.get(1, 0)}`);
// console.log(`Updated B: ${regression.weights.get(0, 0)}`);

const r2 = regression.test(testFeatures, testLabels);

plot({
  y: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});

console.log(`R2 = ${r2}`);
