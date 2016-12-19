import React from "react"


var Prediction = React.createClass({
    render: function() {
        if(this.props.src) {
            return (
                <div className="prediction">
                    <img src={this.props.src} width={this.props.width} height={this.props.height}/>
                </div>
            );
        } else {
            return (
                <div className="prediction">
                </div>
            );
        }
    }
});

export default Prediction

