using NetML;

var ds_train = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_train.csv");
var ds_test = Dataset.load_from_url("https://pjreddie.com/media/files/mnist_test.csv");