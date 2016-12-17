import React, { Component } from 'react'
import { Router, Route, Link, IndexRoute, hashHistory, browserHistory } from 'react-router'
import Header from './Header';


var MainLayout = React.createClass({
    render: function() {
        return (
            <div>
                <Header/>
                {this.props.children}
            </div>
        );
        }
});

const App = MainLayout;
export default App
