import React from "react"
import Alert from 'react-bootstrap/lib/Alert';
import Button from 'react-bootstrap/lib/Button';

class AlertDismissable extends React.Component {

    getInitialState() {
        return {
            alertVisible: true
        };
    }

    render() {
        if (this.props.alertVisible) {
            return (
                <Alert className="alert" bsStyle="danger" onDismiss={this.props.handleAlertDismiss}>
                    <h4>Error</h4>
                    <p>{this.props.alertMessage}</p>
                    <p>
                        <Button onClick={this.props.handleAlertDismiss}>Hide Alert</Button>
                    </p>
                </Alert>
            );
        } else {
            return (<div></div>)
        }
    }

}

export default AlertDismissable
