import React from "react"
import ButtonToolbar from 'react-bootstrap/lib/ButtonToolbar';
import Button from 'react-bootstrap/lib/Button';
import Panel from 'react-bootstrap/lib/Panel';

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
                <Panel header="Controls">
                    <div >
                        <ButtonToolbar>
                            {but}
                            <Button bsStyle="primary">Save</Button>
                            <Button bsStyle="danger" onClick={this.props.stopConvNet}>Stop</Button>
                        </ButtonToolbar>
                    </div>
                </Panel>
            </div>
        );
    }
}

export default Controls