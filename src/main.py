import io
import json

from flask import Flask, request, Response
from flask_cors import CORS
from phase_2.phase2 import multi_lp, support_vm, naive_bayes, set_results_df


""" 
Small backend service for phase 3. 

Implements logic that allows a user to upload their 
own dataset and then runs the various ML algos from 
phase 2. 

Each algorithm returns its visualization component 
as an image that is the rendered on the frontend app
"""

app  = Flask(__name__)
# - enable cors to allow frontend communication
cors = CORS(app)


""" 
Save a figure to a byte stream to send across the webs
"""
def serializeFig(fig):
    out = io.BytesIO()
    fig.savefig(out, format="png")
    return out


""" 
Action that is responsible for receiving a user dataset 
and saving it 
"""
@app.post("/uploadSet")
def uploadSet():
    # - need to delete cached files for subsequent uses
    resp  = {"message": "Success!"}

    if "file" not in request.files:
        resp["message"] = "Error, no file was provided"
        return json.dumps(resp)

    print("uploading data set")
    toUpload = request.files["file"]

    toUpload.save("../user_datasets/" + toUpload.filename)
    #set_results_df("user_datasets/" + toUpload.filename)

    return json.dumps(resp)


""" 
Actions responsible for running phase 2 algorithms and 
returning their corresponding figure's  
"""
@app.get("/getNaiveBayes")
def getNaiveBayes():
    fig = naive_bayes()
    return Response(serializeFig(fig).getvalue(), mimetype="image/png")


@app.get('/getSVM')
def getSVM():
    fig = support_vm()
    return Response(serializeFig(fig).getvalue(), mimetype="image/png")


@app.get('/getMLP')
def getMLP():
    fig = multi_lp()
    return Response(serializeFig(fig).getvalue(), mimetype="image/png")


def main():
    app.run()


if __name__ == "__main__":
    main()
