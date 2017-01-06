import React from "react"
import Prediction from './Prediction';

class Predictions extends React.Component{

    render() {
        let items = [];
        for (let i = this.props.predictions.length - 1; i > 0 ; i--) {
            let prediction = this.props.predictions[i];
            if (prediction.src && prediction.stats) {
                items[i] = <Prediction key={i}
                                       src={prediction.src}
                                       width={prediction.width}
                                       height={prediction.height}
                                       predictions={prediction.stats.activations}
                                       class1={prediction.stats.class1}
                                       class2={prediction.stats.class2}
                                       class1Predication={prediction.stats.class1Predication}
                                       class2Predication={prediction.stats.class2Predication}
                                       expected_prediction={prediction.stats.expectedClass}
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

