import React from "react"
import Button from 'react-bootstrap/lib/Button';
import Panel from 'react-bootstrap/lib/Panel';

class Controls extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            paused: false
        };
    }

    pauseConvNet(){
        this.setState({ paused: true });
    }

    startConvNet(){
        this.setState({ paused: false });
    }

    render() {
        let but;
        if (this.state.paused) {
            but = <Button bsStyle="success" onClick={this.startConvNet.bind(this)}>Go</Button>;
        } else {
            but = <Button bsStyle="danger" onClick={this.pauseConvNet.bind(this)}>Pause</Button>;
        }
        return (

            <div >
                <Panel header="Controls">
                    <div >
                        {but}
                    </div>
                </Panel>
            </div>
        );
    }
}

export default Controls
