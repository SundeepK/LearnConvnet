import * as d3 from "d3";
'use strict';

class LossGraph {

    constructor(height, width, element) {
        this.height = height;
        this.width = width;
        let margin = {top: 5, right: 20, bottom: 10, left: 50};
        this.width = width - margin.left - margin.right;
        this.height = height - margin.top - margin.bottom;

        // set the ranges
        var x = d3.scaleLinear().range([0, width]);
        var y = d3.scaleLinear().range([height, 0]);

        // define the line
        this.valueline = d3.line()
            .x(function(d) { return x(d.count); })
            .y(function(d) { return y(d.stats.cost_loss); });

        this.x = x;
        this.y = y;
        // append the svg object to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        this.svg = d3.select(element).append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        // Add the X Axis
        this.svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + this.height + ")")
            .call(d3.axisBottom(this.x));

        // Add the Y Axis
        this.svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(0," + "-15" + ")")
            .call(d3.axisLeft(this.y));

    }

    update(data){
        this.x.domain(d3.extent(data, function (d) {
            return d.count;
        }));

        this.y.domain([0, d3.max(data, function (d) {
            return d.stats.cost_loss;
        })]);

        var d = this.svg.selectAll(".line")
            .data(data);

        d.remove()
            .exit();

        d.enter()
            .append("path")
            .datum(data)
            .attr("class", "line")
            .attr("d", this.valueline);

        this.svg.select(".x") // change the x axis
            .call(this.x);

        this.svg.select(".y") // change the y axis
            .call(this.y);


    }
}
export default LossGraph;


