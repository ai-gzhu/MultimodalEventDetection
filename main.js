var w2v = require("word2vector");
var modelFile = "./GoogleNews-vectors-negative300.bin";
w2v.load( modelFile );
w2v.similarity("gun", "violence")
