import React from "react"
import ButtonToolbar from 'react-bootstrap/lib/ButtonToolbar';
import Button from 'react-bootstrap/lib/Button';

class Controls extends React.Component {

    constructor(props) {
        super(props);
    }

    render() {
        let but;
        if (!this.props.canPause) {
            but = <Button bsStyle="success" onClick={this.props.startConvNet}>Go</Button>;
        } else {
            but = <Button bsStyle="danger" onClick={this.props.pauseConvNet}>Pause</Button>;
        }
        return (
            <div >
                <div >
                    <ButtonToolbar>
                        {but}
                        <Button bsStyle="primary" onClick={this.props.saveConvNet}>Save</Button>
                        <Button bsStyle="danger" onClick={this.props.stopConvNet}>Stop</Button>
                    </ButtonToolbar>
                </div>
            </div>
        );
    }
}

export default Controls
