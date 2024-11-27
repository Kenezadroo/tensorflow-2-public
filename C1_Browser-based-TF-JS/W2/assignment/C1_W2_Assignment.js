import {FMnistData} from './fashion-data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;

function getModel() {
    
    // In the space below create a convolutional neural network that can classify the 
    // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
    // neural network should only use the following layers: conv2d, maxPooling2d,
    // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
    // should have 10 units and a softmax activation function. You are free to use as
    // many layers, filters, and neurons as you like.  
    // HINT: Take a look at the MNIST example.
    model = tf.sequential();
    
    // Adding layers to the model
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1], // input is a 28x28 grayscale image
        filters: 32,             // number of filters (feature maps)
        kernelSize: 3,           // size of the convolution kernel
        activation: 'relu'       // ReLU activation function for non-linearity
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],        // max pooling layer with 2x2 pool size
        strides: [2, 2]          // strides of 2 for downsampling
    }));

    model.add(tf.layers.conv2d({
        filters: 64,             // number of filters
        kernelSize: 3,           // kernel size
        activation: 'relu'       // ReLU activation
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],        // max pooling with 2x2 pool size
        strides: [2, 2]          // strides of 2
    }));

    model.add(tf.layers.flatten());  // Flatten the output from 2D to 1D

    model.add(tf.layers.dense({
        units: 128,              // fully connected layer with 128 units
        activation: 'relu'       // ReLU activation
    }));

    model.add(tf.layers.dense({
        units: 10,               // output layer with 10 units (one for each class)
        activation: 'softmax'    // softmax activation for multi-class classification
    }));

    // Compile the model
    model.compile({
        loss: 'categoricalCrossentropy',    // loss function for multi-class classification
        optimizer: tf.train.adam(),         // Adam optimizer
        metrics: ['acc']                    // track accuracy
    });
    
    // Compile the model using the categoricalCrossentropy loss,
    // the tf.train.adam() optimizer, and `acc` for your metrics.
    model.compile({
        loss: 'categoricalCrossentropy',    // loss function for multi-class classification
        optimizer: tf.train.adam(),         // Adam optimizer
        metrics: ['acc']                    // track accuracy
    });
    
    return model;
}

async function train(model, data) {
        
    // Set the following metrics for the callback: 'loss', 'val_loss', 'acc', 'val_acc'.
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];    

        
    // Create the container for the callback. Set the name to 'Model Training' and 
    // use a height of 1000px for the styles. 
    const container = document.getElementById('container');
    container.style.height = '1000px';   
    
    
    // Use tfvis.show.fitCallbacks() to setup the callbacks. 
    // Use the container and metrics defined above as the parameters.
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 6000;
    const TEST_DATA_SIZE = 1000;
    
    // Get the training batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        const xs = d.xs.reshape([d.xs.shape[0], 28, 28, 1]);
        const ys = d.ys;
        return [xs, ys];
    });

    
    // Get the testing batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        const xs = d.xs.reshape([d.xs.shape[0], 28, 28, 1]);
        const ys = d.ys;
        return [xs, ys];
    });

    
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

function setPosition(e){
    pos.x = e.clientX-100;
    pos.y = e.clientY-100;
}
    
function draw(e) {
    if(e.buttons!=1) return;
    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,280,280);
}
    
function save() {
    var raw = tf.browser.fromPixels(rawImage,1);
    var resized = tf.image.resizeBilinear(raw, [28,28]);
    var tensor = resized.expandDims(0);
    
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
    var classNames = ["T-shirt/top", "Trouser", "Pullover", 
                      "Dress", "Coat", "Sandal", "Shirt",
                      "Sneaker",  "Bag", "Ankle boot"];
            
            
    alert(classNames[pIndex]);
}
    
function init() {
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasimg');
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,280,280);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    saveButton = document.getElementById('sb');
    saveButton.addEventListener("click", save);
    clearButton = document.getElementById('cb');
    clearButton.addEventListener("click", erase);
}


async function run() {
    const data = new FMnistData();
    await data.load();
    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    await train(model, data);
    await model.save('downloads://my_model');
    init();
    alert("Training is done, try classifying your drawings!");
}

document.addEventListener('DOMContentLoaded', run);



