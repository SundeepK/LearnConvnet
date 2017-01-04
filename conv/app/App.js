import React, {Component} from 'react'
import Panel from 'react-bootstrap/lib/Panel';
import ListGroup from 'react-bootstrap/lib/ListGroup';
import ListGroupItem from 'react-bootstrap/lib/ListGroupItem';
import {Router, Route, Link, IndexRoute, hashHistory, browserHistory} from 'react-router'
import Header from './Header';
import Graph from './Graph';
import Controls from './Controls';
import Stats from './Stats';
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

const uid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {let r = Math.random()*16|0,v=c=='x'?r:r&0x3|0x8;return v.toString(16);});

class MainLayout extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            canPause: false,
            clearGraph: false,
            predictions: this.emptyObjectList()
        };
        this.emptyObjectList = this.emptyObjectList.bind(this);
        this.getRunningAvgClassificationLoss = this.getRunningAvgClassificationLoss.bind(this);
        this.count = 0;
    }

    emptyObjectList(){
        let arr = [];
        for (let i = 0; i < 265; i++) {
            arr[i] = {};
        }
        return arr;
    }

    componentDidMount() {
        this.ws = new WebSocket('ws://localhost:8888/ws');
        this.ws.binaryType = "arraybuffer";
        this.ws.onopen = function () {
        };
        this.ws.onmessage = this.onMessage.bind(this);
        this.addLatestPrediction = this.addLatestPrediction.bind(this);
        this.addLatestPredictionImg = this.addLatestPredictionImg.bind(this);
    }

    onMessage(evt) {
        if (typeof evt.data === "string") {
            this.addLatestPrediction(evt);
        } else {
            this.addLatestPredictionImg(evt);
        }
    }

    addLatestPredictionImg(evt) {
        let imageWidth = 32, imageHeight = 32;
        let blob = new Blob([(evt.data)], {type: 'image/jpeg'});
        let url = (window.URL || window.webkitURL).createObjectURL(blob);
        let image = {};
        image.width = imageWidth;
        image.height = imageHeight;
        image.src = url;

        const predictions = this.state.predictions;
        if (predictions.length >= 265) {
            predictions.pop();
        }
        predictions.unshift(image);
        this.setState({predictions: predictions});
    }

    addLatestPrediction(evt) {
        let stats = JSON.parse(evt.data);
        const predictions = this.state.predictions;
        let maxAct = 0;
        let max = 0;
        let secondMax = 0;
        for (let i = 0; i < stats.activations.length; i++) {
            if (stats.activations[i] > maxAct) {
                maxAct = stats.activations[i];
                secondMax = max;
                max = i;
            }
        }
        this.count++;
        predictions[0].stats = stats;
        predictions[0].stats.class1 = classes[max];
        predictions[0].stats.class2 = classes[secondMax];
        predictions[0].stats.class1Predication = (stats.activations[max] * 100).toFixed(2);
        predictions[0].stats.class2Predication = (stats.activations[secondMax] * 100).toFixed(2);
        predictions[0].count = this.count;
        this.setState({predictions: predictions});
    }

    pauseConvNet() {
        this.setState({canPause: false});
        this.ws.send(JSON.stringify({ pause: true, id: uid}))
    }

    startConvNet() {
        this.setState({canPause: true});
        this.ws.send(JSON.stringify({ pause: false, id: uid }))
    }

    stopConvNet() {
        this.setState({stop: true, canPause: false, predictions: this.emptyObjectList()});
        this.count = 0;
        this.ws.send(JSON.stringify({ stop: true, id: uid }))
    }

    getRunningAvgClassificationLoss(){
        let predictions = this.state.predictions;
        let totalClassLoss = 0;
        let total= 0 ;
        for(let i = 0; i < predictions.length; i++){
            if (predictions[i].hasOwnProperty("stats")) {
                totalClassLoss += predictions[i].stats.cost_loss;
                total++;
            }
        }
        if (totalClassLoss == 0) {
            return 0;
        }
        return (totalClassLoss / total);
    }

    render() {
        return (
            <div>
                <Header/>
                <div className="container-main">
                    <div className="content">
                        <div className="controls">
                            <Panel collapsible defaultExpanded header="Controls & stats">
                                <ListGroup fill>
                                    <ListGroupItem>
                                        <Controls canPause={this.state.canPause}
                                                  pauseConvNet={this.pauseConvNet.bind(this)}
                                                  stopConvNet={this.stopConvNet.bind(this)}
                                                  startConvNet={this.startConvNet.bind(this)}/>
                                    </ListGroupItem>
                                    <ListGroupItem>
                                        <Stats avgClassificationLoss={this.getRunningAvgClassificationLoss()} totalExamples={this.count}/>
                                    </ListGroupItem>
                                </ListGroup>
                            </Panel>
                        </div>
                        <Graph key="twograph" data={this.state.predictions}/>
                    </div>
                    <Predictions predictions={this.state.predictions}/>
                </div>

            </div>
        );
    }
}

const App = MainLayout;
export default App
