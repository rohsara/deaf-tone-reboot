let model, video, keypoints, predictions=[]; 
let fft, osc, label;
// Create a KNN classifier
const classifier = knnClassifier.create();

const LABELS_MAP = {
	'Do': [262, 'C4'],
	'Re': [294, 'D4'],
	'Mi': [330, 'E4'],
	'Fa': [350, 'F4'],
	'Sol': [392, 'G4'], 
	'La': [440, 'A4'],
	'Ti': [494, 'B4'],
	'Do5': [523, 'C5']
};

function preload() {
	video = createCapture(VIDEO, () => {
	  loadHandTrackingModel();
	});
	video.hide();
	// Create the UI buttons
	createButtons();
}

function setup() {
	const canvas = createCanvas(480, 360);
	canvas.parent('canvasContainer');
	
	osc = new p5.Oscillator('sine');
	osc.amp(0.1);
	// osc.start();

	fft = new p5.FFT();
}

async function loadHandTrackingModel() {
	// Load the MediaPipe handpose model.
	model = await handpose.load();
	select('#status').html('Hand Tracking Model Loaded')
	predictHand();
}

function draw() {
	background(255);
	if (model) image(video, 0, 0);
	filter(GRAY);
	if (predictions.length > 0) {
		// We can call both functions to draw all keypoints and the skeletons
		drawKeypoints();
		drawSkeleton();

		let waveform = fft.waveform();

		noFill();
		beginShape();
		stroke(20);
		for (let i = 0; i < waveform.length; i++){
			let x = map(i, 0, waveform.length, 0, width);
			let y = map(waveform[i], -1, 1, 0, height);
			vertex(x,y);
		}
		endShape();

		const waveHeight = label ? LABELS_MAP[label][0] : 0;
  
		//VCR colors
		waveVisual(255,255,255, 40, waveHeight); 
		waveVisual(249, 251, 0, 70, waveHeight); // yellow
		waveVisual(6, 253, 255, 50, waveHeight); // cyon
		waveVisual(12, 254, 0, 50, waveHeight); // green
		waveVisual(253, 4, 251, 50, waveHeight); // magenta
		waveVisual(251, 2, 4, 100, waveHeight); // red
		waveVisual(21, 2, 252, 150, waveHeight); // blue
	}
}

function waveVisual(r, b, g, a, waveHeight) {
	var waveform = fft.waveform();
  
	noFill();
	beginShape();
	stroke(r, b, g, a);
	strokeWeight(1);
	
	const range = 200 / waveHeight;
  
	for (var i = 0; i < waveform.length; i++) {
	  var x = map(i, 0, waveform.length, 0, width);
	  var y = map(waveform[i], -range, range, 0, height);
	  vertex(x, y);
	  vertex(x + 2, y + 2);
	  // vertex(center1, center2);
	}
	endShape();
  }

async function predictHand() {
	// Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a
	// hand prediction from the MediaPipe graph.
	predictions = await model.estimateHands(video.elt);
  
	setTimeout(() => predictHand(), 200);
}

// Add the current hand tracking data to the classifier
function addExample(label) {
	if (predictions.length > 0) {
	  const features = predictions[0].landmarks;
	  const tensors = tf.tensor(features)
	  // Add an example with a label to the classifier
	  classifier.addExample(tensors, label);
	  updateCounts();
	} else {
	  console.log('No gesture is detected')
	}
}

// Predict the current frame.
async function classify() {
	// Get the total number of labels from classifier
	const numLabels = classifier.getNumClasses();
	if (numLabels <= 0) {
	  console.error('There is no examples in any label');
	  osc.stop();
	  return;
	}
	if (predictions.length > 0) {
	    osc.start();
		const results = await classifier.predictClass(tf.tensor(predictions[0].landmarks));
		if (results.confidences) {
			const confidences = results.confidences;
			// Change the freq of osc
			label = results.label;
			const freq = LABELS_MAP[label][0];
			osc.freq(freq);
			// result.label is the label that has the highest confidence
			if (results.label) {
				select('#result').html(results.label);
				select('#confidence').html(`${confidences[results.label] * 100} %`);
			}

			select('#confidenceDo').html(`${confidences['Do'] ? confidences['Do'] * 100 : 0} %`);
			select('#confidenceRe').html(`${confidences['Re'] ? confidences['Re'] * 100 : 0} %`);
			select('#confidenceMi').html(`${confidences['Mi'] ? confidences['Mi'] * 100 : 0} %`);
			select('#confidenceFa').html(`${confidences['Fa'] ? confidences['Fa'] * 100 : 0} %`);
			select('#confidenceSol').html(`${confidences['Sol'] ? confidences['Sol'] * 100 : 0} %`);
			select('#confidenceLa').html(`${confidences['La'] ? confidences['La'] * 100 : 0} %`);
			select('#confidenceTi').html(`${confidences['Ti'] ? confidences['Ti'] * 100 : 0} %`);
			select('#confidenceDo5').html(`${confidences['Do5'] ? confidences['Do5'] * 100 : 0} %`);
		}
		classify();
  	} else {
    	setTimeout(() => classify(), 1000);
  	}
}

// Update the example count for each label	
function updateCounts() {
	const counts = classifier.getClassExampleCount();

	select('#exampleDo').html(counts['Do'] || 0);
	select('#exampleRe').html(counts['Re'] || 0);
	select('#exampleMi').html(counts['Mi'] || 0);
	select('#exampleFa').html(counts['Fa'] || 0);
	select('#exampleSol').html(counts['Sol'] || 0);
	select('#exampleLa').html(counts['La'] || 0);
	select('#exampleTi').html(counts['Ti'] || 0);
	select('#exampleDo5').html(counts['Do5'] || 0);
}

// Clear the examples in one label
function clearLabel(label) {
	classifier.clearClass(label);
	updateCounts();
}
  
// Clear all the examples in all labels
function clearAllLabels() {
	classifier.clearAllClasses();
	updateCounts();
}
  
// A util function to create UI buttons
function createButtons() {
	// When the A button is pressed, add the current frame
	// from the video with a label of "do" to the classifier
	buttonC = select('#addClassDo');
	buttonC.mousePressed(function(){
		addExample('Do');
	}); 
	
	buttonD = select('#addClassRe');
	buttonD.mousePressed(function(){
		addExample('Re');
	}); 
	
	buttonE = select('#addClassMi');
	buttonE.mousePressed(function(){
		addExample('Mi');
	}); 
	
	buttonF = select('#addClassFa');
	buttonF.mousePressed(function(){
		addExample('Fa');
	}); 
	
	buttonG = select('#addClassSol');
	buttonG.mousePressed(function(){
		addExample('Sol');
	}); 
	
	buttonA = select('#addClassLa');
	buttonA.mousePressed(function(){
		addExample('La');
	}); 

	buttonB = select('#addClassTi');
	buttonB.mousePressed(function(){
		addExample('Ti');
	}); 

	buttonC5 = select('#addClassDo5');
	buttonC5.mousePressed(function(){
		addExample('Do5');
	}); 
	
	// Reset Buttons
	resetBtnC = select('#resetDo');
	resetBtnC.mousePressed(function(){
		addExample('Do');
	}); 
	
	resetBtnD = select('#resetRe');
	resetBtnD.mousePressed(function(){
		addExample('Re');
	}); 
	
	resetBtnE = select('#resetMi');
	resetBtnE.mousePressed(function(){
		addExample('Mi');
	}); 
	
	resetBtnF = select('#resetFa');
	resetBtnF.mousePressed(function(){
		addExample('Fa');
	}); 
	
	resetBtnG = select('#resetSol');
	resetBtnG.mousePressed(function(){
		addExample('Sol');
	}); 
	
	resetBtnA = select('#resetLa');
	resetBtnA.mousePressed(function(){
		addExample('La');
	}); 

	resetBtnB = select('#resetTi');
	resetBtnB.mousePressed(function(){
		addExample('Ti');
	}); 

	resetBtnC5 = select('#resetDo5');
	resetBtnC5.mousePressed(function(){
		addExample('Do5');
	}); 

	// Predict Button
	buttonPredict = select('#buttonPredict');
	buttonPredict.mousePressed(classify);
	
	// Clear all classes button
	buttonClearAll = select('#clearAll');
	buttonClearAll.mousePressed(clearAllLabels);
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints()  {
	let prediction = predictions[0];
	for (let j = 0; j < prediction.landmarks.length; j++) {
	  let keypoint = prediction.landmarks[j];
	  fill(255, 0, 0);
	  noStroke();
	  ellipse(keypoint[0], keypoint[1], 10, 10);
	}
}
  
  // A function to draw the skeletons
function drawSkeleton() {
	let annotations = predictions[0].annotations;
	stroke(255, 0, 0);
	for (let j = 0; j < annotations.thumb.length - 1; j++) {
	  line(annotations.thumb[j][0], annotations.thumb[j][1], annotations.thumb[j + 1][0], annotations.thumb[j + 1][1]);
	}
	for (let j = 0; j < annotations.indexFinger.length - 1; j++) {
	  line(annotations.indexFinger[j][0], annotations.indexFinger[j][1], annotations.indexFinger[j + 1][0], annotations.indexFinger[j + 1][1]);
	}
	for (let j = 0; j < annotations.middleFinger.length - 1; j++) {
	  line(annotations.middleFinger[j][0], annotations.middleFinger[j][1], annotations.middleFinger[j + 1][0], annotations.middleFinger[j + 1][1]);
	}
	for (let j = 0; j < annotations.ringFinger.length - 1; j++) {
	  line(annotations.ringFinger[j][0], annotations.ringFinger[j][1], annotations.ringFinger[j + 1][0], annotations.ringFinger[j + 1][1]);
	}
	for (let j = 0; j < annotations.pinky.length - 1; j++) {
	  line(annotations.pinky[j][0], annotations.pinky[j][1], annotations.pinky[j + 1][0], annotations.pinky[j + 1][1]);
	}
  
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.thumb[0][0], annotations.thumb[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.indexFinger[0][0], annotations.indexFinger[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.middleFinger[0][0], annotations.middleFinger[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.ringFinger[0][0], annotations.ringFinger[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.pinky[0][0], annotations.pinky[0][1]);
}  