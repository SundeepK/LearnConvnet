import React from "react"
import Prediction from './Prediction';


var Predictions = React.createClass({
    render: function() {
        let items = [];
        for (let i = 0; i < 253; i++) {
            items[i] = <Prediction/>
        }

        return (
            <div className="predictions">
                {items}
            </div>
        );
    }
});

export default Predictions

