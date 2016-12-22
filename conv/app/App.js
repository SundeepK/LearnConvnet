import React, {Component} from 'react'
import {Router, Route, Link, IndexRoute, hashHistory, browserHistory} from 'react-router'
import Header from './Header';
import Graph from './Graph';
import Controls from './Controls';
import Predictions from './Predictions';

const classes = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
};

class MainLayout extends React.Component {

    constructor(props) {
        super(props);
        let pred = [];
        for (var i = 0; i < 256; i++) {
            pred[i] = {};
        }
        this.state = {
            canPause: false,
            predictions: pred
        };
    }

    componentDidMount() {
        this.ws = new WebSocket('ws://localhost:8888/ws');
        this.ws.binaryType = "arraybuffer";
        this.ws.onopen = function () {
        };
        this.ws.onmessage = this.onMessage.bind(this)
    }

    onMessage(evt) {
        if (typeof evt.data === "string") {
            let stats = JSON.parse(evt.data);
            const predictions = this.state.predictions;
            var maxAct = 0;
            var max = 0;
            var secondMax = 0;
            for (var i = 0; i < stats.activations.length; i++) {
                if (stats.activations[i] > maxAct) {
                    maxAct = stats.activations[i];
                    secondMax = max;
                    max = i;
                }
            }
            predictions[0].stats = stats;
            predictions[0].stats.class1 = classes[max];
            predictions[0].stats.class2 = classes[secondMax];
            predictions[0].stats.class1Predication = (stats.activations[max] * 100).toFixed(2);
            predictions[0].stats.class2Predication = (stats.activations[secondMax] * 100).toFixed(2);
        } else {
            var imageWidth = 32, imageHeight = 32;
            var blob = new Blob([(evt.data)], {type: 'image/jpeg'});
            var url = (window.URL || window.webkitURL).createObjectURL(blob);
            var image = {};
            image.width = imageWidth;
            image.height = imageHeight;
            image.src = url;

            const predictions = this.state.predictions;
            if (predictions.length >= 253) {
                predictions.pop();
            }
            predictions.unshift(image);
            this.setState({predictions: predictions});
        }
    }

    pauseConvNet() {
        console.log({canPause: false})
        this.setState({canPause: false});
        this.ws.send(JSON.stringify({pause: true}))
    }

    startConvNet() {
        console.log({canPause: true})
        this.setState({canPause: true});
        this.ws.send(JSON.stringify({pause: false}))
    }

    render() {
        return (
            <div>
                <Header/>
                <div className="container-main">
                    <div className="controls">
                        <Controls canPause={this.state.canPause} pauseConvNet={this.pauseConvNet.bind(this)}
                                  startConvNet={this.startConvNet.bind(this)}/>
                    </div>
                    <div className="content">
                        <Graph/><Graph/>
                    </div>
                    <Predictions predictions={this.state.predictions}/>
                </div>

            </div>
        );
    }
}

const App = MainLayout;
export default App
