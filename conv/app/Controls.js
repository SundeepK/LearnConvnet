import React from "react"
import ButtonToolbar from 'react-bootstrap/lib/ButtonToolbar';
import Button from 'react-bootstrap/lib/Button';
import FormControl from 'react-bootstrap/lib/FormControl';

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
                        <Button bsStyle="primary" onClick={this.props.saveConvNet}>Save as json</Button>
                        <Button bsStyle="danger" onClick={this.props.stopConvNet}>Stop</Button>
                    </ButtonToolbar>
                </div>
                <ButtonToolbar className="input-control">
                    <Button bsStyle="primary" onClick={this.props.loadConvNet}>Load CNN</Button>
                    <FormControl  ref='fileUpload' className="input-file" type="file" onChange={this.props.onFileSelect}/>
                </ButtonToolbar>
            </div>
        );
    }
}

export default Controls
