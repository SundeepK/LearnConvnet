import React from "react"
import ReactDOM from "react-dom"
import LossGraph from './lossGraph';

class Graph extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            predictions:[]
        };
        this.setUpLossGraph = this.setUpLossGraph.bind(this);
        this.updateLossGraph = this.updateLossGraph.bind(this);
        this.lossGraph = {};
    }

    setUpLossGraph(el){
        this.lossGraph = new LossGraph(300, 750, el);
    }

    updateLossGraph(data){
        this.lossGraph.update(data);
    }

    componentDidMount() {
        var el = ReactDOM.findDOMNode(this);
        this.setUpLossGraph(el);
    }

    componentDidUpdate(prevProps, prevState) {
        let itemsToUpdate = prevProps.data.filter(function(d) {
            return d.hasOwnProperty("count");
        }).map(function(itr) {
            return {count: itr.count, cost_loss: parseFloat(itr.stats.cost_loss)}
        });
        if (itemsToUpdate.length > 0) {
            this.updateLossGraph(itemsToUpdate)
        }
    }

    componentWillUnmount() {
        var el = this.getDOMNode();
    }


    render() {
        return (
        <div className="graph"></div>
        );
    }
}

export default Graph

