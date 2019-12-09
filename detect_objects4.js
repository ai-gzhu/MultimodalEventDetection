console.log("Now detecting event "+process.argv[2]+" in video file: "+process.argv[3]);


var w2v = require("word2vector");
var w2vVectors = "./GoogleNews-vectors-negative300.bin";
w2v.load( w2vVectors );

const fs = require("fs");
const path = require("path");
const { cv, runVideoDetection } = require("./utils");
// replace with path where you unzipped inception model
const inceptionModelPath = './inception5h'

const modelFile = path.resolve(inceptionModelPath, 'tensorflow_inception_graph.pb');
const classNamesFile = path.resolve(inceptionModelPath, 'imagenet_comp_graph_label_strings.txt');
if (!fs.existsSync(modelFile) || !fs.existsSync(classNamesFile)) {
  console.log('exiting: could not find inception model');
  console.log('download the model from: https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip');
  return;
}

// read classNames and store them in an array
const classNames = fs.readFileSync(classNamesFile).toString().split("\n");

// initialize tensorflow inception model from modelFile
const net = cv.readNetFromTensorflow(modelFile);

var query = w2v.getVector(process.argv[2]);

//var query = w2v.add(w2v.getVector("gun"), w2v.getVector("violence"));

//var query = w2v.add(w2v.getVector("gun"), w2v.getVector("violence"));
function cosinesim(A,B){
    var dotproduct=0;
    var mA=0;
    var mB=0;
    for(i = 0; i < A.length; i++){ // here you missed the i++
        dotproduct += (A[i] * B[i]);
        mA += (A[i]*A[i]);
        mB += (B[i]*B[i]);
    }
    mA = Math.sqrt(mA);
    mB = Math.sqrt(mB);
    var similarity = (dotproduct)/((mA)*(mB)) // here you needed extra brackets
    return similarity;
}

const classifyImg = (img) => {
  // inception model works with 224 x 224 images, so we resize
  // our input images and pad the image with white pixels to
  // make the images have the same width and height
  const maxImgDim = 224;
  const white = new cv.Vec(255, 255, 255);
  const imgResized = img.resizeToMax(maxImgDim).padToSquare(white);

  // network accepts blobs as input
  const inputBlob = cv.blobFromImage(imgResized);
  net.setInput(inputBlob);

  // forward pass input through entire network, will return
  // classification result as 1xN Mat with confidences of each class
  const outputBlob = net.forward();

  // find all labels with a minimum confidence
  const minConfidence = 0.05;
  const locations =
    outputBlob
      .threshold(minConfidence, 1, cv.THRESH_BINARY)
      .convertTo(cv.CV_8U)
      .findNonZero();

  const results =
    locations.map(pt => ({
      confidence: outputBlob.at(0, pt.x),
      className: classNames[pt.x]
    }))
      // sort result by confidence
      //.sort((r0, r1) => r1.confidence-r0.confidence)
//      .map(res => `${res.className} (${res.confidence})`);

  //results.forEach(p => console.log(p));
  if(results)
  {
  	var similarity = 0.;
  	var sum = 0.;
  	
  	var top_similar_value = 0;
  	var top_similar_class = "";
  	
  	var top_confidence_value = 0;
  	var top_confidence_class = "";
  	
  	var top_product_value = 0;
  	var top_product_class = 0;
  	
  	results.map( p => {
  		var vector = w2v.getVector(p.className);
  		try
  		{
  			var s = Math.min(1., w2v.similarity(query, vector));//cosinesim(query, vector);//;
  			if(s > top_similar_value)
  			{
  				top_similar_value = s;
  				top_similar_class = p.className;
  			}
  			
  			if(p.confidence > top_confidence_value)
  			{
  				top_confidence_value = p.confidence;
  				top_confidence_class = p.className;
  			}
  			
  			if(p.confidence*s > top_product_value)
  			{
  				top_product_value = p.confidence*s;
  				top_product_class = p.className;
  			}
  			
	  		similarity += s*p.confidence;
	  		sum += p.confidence;
		}
		catch(e)
		{
			//TODO: Fix
		}
  	});
  	var bars = 100;
  	var bargraph = ""
  	for(var i = 0; i < similarity/sum*bars; i++)
  		bargraph += "|";
  	for(var i = 0; i < (1.-similarity/sum)*bars; i++)
  		bargraph += "-";
  		
  	console.log(bargraph+top_product_class);//+" / "+" / "+top_similar_class+" / "+top_confidence_class);
  }
  cv.imshow("Temsorflow Object Detection", img.resizeToMax(2048));
  return results;
}


// set webcam port
const webcamPort = process.argv[3];

runVideoDetection(webcamPort, classifyImg);
