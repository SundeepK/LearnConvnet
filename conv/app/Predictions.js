import React from "react"
import Prediction from './Prediction';

const classes = {0: 'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'};
class Predictions extends React.Component{

    constructor(props) {
        super(props);
        let pred = [];
        for (var i = 0; i < 256; i++) {
            pred[i] = {};
        }
        this.state = {
            predictions: pred
        };
    }

    componentDidMount() {
        var ws = new WebSocket('ws://localhost:8888/ws');
        ws.binaryType = "arraybuffer";
        ws.onopen = function(){
        };
        ws.onmessage = this.onMessage.bind(this)
    }

    onMessage(evt){
        if (typeof evt.data === "string") {
            let stats = JSON.parse(evt.data);
            const predictions = this.state.predictions;
            var maxAct = 0
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
        } else {
            var imageWidth = 32, imageHeight = 32;
            var blob = new Blob([(evt.data)], {type: 'image/jpeg'});
            var url = (window.URL || window.webkitURL).createObjectURL(blob);
            var image = {};
            image.width = imageWidth;
            image.height = imageHeight;
            image.src = url;

            const predictions = this.state.predictions;
            if (predictions.length >= 253){
                predictions.pop();
            }
            predictions.unshift(image);
            this.setState({ predictions: predictions });
        }
    }

    render() {
        let items = [];
        for (let i = this.state.predictions.length - 1; i > 0 ; i--) {
            let prediction = this.state.predictions[i];
            if (prediction.src && prediction.stats) {
                items[i] = <Prediction key={i}
                                       src={prediction.src}
                                       width={prediction.width}
                                       height={prediction.height}
                                       predictions={prediction.stats.activations}
                                       class1={prediction.stats.class1}
                                       class2={prediction.stats.class2}
                />
            } else {
                items[i] = <Prediction key={i} />
            }
        }
        return (
            <div className="predictions">
                {items}
            </div>
        );
    }
}

export default Predictions

