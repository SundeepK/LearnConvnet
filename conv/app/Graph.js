import React from "react"
import ReactDOM from "react-dom"
import LossGraph from './lossGraph';

const MAX_POINTS_BEFORE_TAKING_AVG = 50;

class Graph extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            predictions: []
        };
        this.costAvgs = {};
        this.keys = [];
        this.setUpLossGraph = this.setUpLossGraph.bind(this);
        this.updateLossGraph = this.updateLossGraph.bind(this);
        this.getRunningAvgClassificationLoss = this.getRunningAvgClassificationLoss.bind(this);
        this.getFromAvgs = this.getFromAvgs.bind(this);
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

        // re-init hash and arrays because first item seen
        if(itemsToUpdate.length == 0 && this.keys.length > 0){
            this.costAvgs = {};
            this.keys = [];
            this.lossGraph.clear();
        }

        if (itemsToUpdate.length > 0 && itemsToUpdate[0].count % MAX_POINTS_BEFORE_TAKING_AVG == 0) {
                  if (!this.costAvgs.hasOwnProperty(itemsToUpdate[0].count)) {
                    this.costAvgs[itemsToUpdate[0].count] = {
                        count: itemsToUpdate[0].count,
                        cost_loss: this.getRunningAvgClassificationLoss(itemsToUpdate)
                    };
                    this.keys.push(itemsToUpdate[0].count);
            }
            let data = this.keys.map(this.getFromAvgs);
            this.updateLossGraph(data)
        }
    }

    getFromAvgs(i){
        return this.costAvgs[i]
    }

    getRunningAvgClassificationLoss(data){
        let totalClassLoss = 0;
        let total = 0;
        for(let i = 0; i < data.length; i++){
            if (data[i].hasOwnProperty("count")) {
                totalClassLoss += data[i].cost_loss;
                total++;
            }
        }
        if (totalClassLoss == 0 || total == 0 ) {
            return 0;
        }
        return (totalClassLoss / total);
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

