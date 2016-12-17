import React, { Component } from 'react'
import { Router, Route, Link, IndexRoute, hashHistory, browserHistory } from 'react-router'
import Header from './Header';
import Graph from './Graph';
import Predictions from './Predictions';


var MainLayout = React.createClass({
    render: function () {
        return (
            <div>
                <div >
                    <Header/>
                </div>
                <div className="container">
                    <div className="content">
                        <Graph/><Graph/>
                        {this.props.children}
                    </div>
                    <Predictions/>
                </div>
            </div>
        );
    }
});

const App = MainLayout;
export default App
