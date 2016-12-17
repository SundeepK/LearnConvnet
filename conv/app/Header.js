import React from "react"
import { Navbar, Nav, NavItem, NavDropdown, MenuItem } from 'react-bootstrap';


var Header = React.createClass({
    render: function() {
        return (
            <Navbar className="navbar navbar-default navbar-static-top">
                <Navbar.Header >
                    <Navbar.Brand>
                        <a href="#">Learn Convnet</a>
                    </Navbar.Brand>
                </Navbar.Header>
            </Navbar>
        );
    }
});

export default Header

