import React from "react"
import Prediction from './Prediction';


class Predictions extends React.Component{

    constructor(props) {
        super(props);
        let pred = [];
        for (var i = 0; i < 253; i++) {
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
            console.log(evt.data)
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
            let img = this.state.predictions[i];
            if (img.src) {
                items[i] = <Prediction key={i} src={img.src} width={img.width} height={img.height}/>
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

