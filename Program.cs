using NetML;
using NetML.ML;

var ds_train = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
var ds_test = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");

var mlp = new Network([784, 128, 10]);
var trainer = new Trainer(mlp);

trainer.train(ds_train, 0.0085f, 5);