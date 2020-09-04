# Judge a book by its cover

Based off https://github.com/mtobeiyf/keras-flask-deploy-webapp

#### To run locally:

```shell
# 1. Install Python packages
$ pip install -r requirements.txt

# 2. Run
$ python app.py
```

------------------

#### To update models:

Place the trained `.h5` file saved by `model.save()` under models directory.

Check the [commented code](https://github.com/mtobeiyf/keras-flask-deploy-webapp/blob/master/app.py#L37) in app.py.

#### To use another pre-trained model

See [Keras applications](https://keras.io/applications/) for more available models such as DenseNet, MobilNet, NASNet, etc.

Check [this section](https://github.com/mtobeiyf/keras-flask-deploy-webapp/blob/master/app.py#L26) in app.py.

#### UI Modification

Modify files in `templates` and `static` directory.

`index.html` for the UI and `main.js` for all the behaviors.

