import io
import json

from flask import Flask, request, Response
from flask_cors import CORS
from phase_2.phase2 import multi_lp, support_vm, naive_bayes


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

    to_save = f"../user_datasets/{toUpload.filename}"
    toUpload.save(to_save)

    return json.dumps(resp)


""" 
Actions responsible for running phase 2 algorithms and 
returning their corresponding figure's  
"""
@app.get("/getNaiveBayes")
def getNaiveBayes():
    f_name = request.args.get("f_name")
    fig = naive_bayes(f"../user_datasets/{f_name}")
    return Response(serializeFig(fig).getvalue(), mimetype="image/png")


@app.get('/getSVM')
def getSVM():
    f_name = request.args.get("f_name")
    fig = support_vm(f"../user_datasets/{f_name}")
    return Response(serializeFig(fig).getvalue(), mimetype="image/png")


@app.get('/getMLP')
def getMLP():
    f_name = request.args.get("f_name")
    fig = multi_lp(f"../user_datasets/{f_name}")
    return Response(serializeFig(fig).getvalue(), mimetype="image/png")


def main():
    app.run()


if __name__ == "__main__":
    main()
