import React from "react"
import Table from 'react-bootstrap/lib/Table';

class Stats extends React.Component {

    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div >
                Last 200 examples
                <div >
                    <Table responsive>
                        <tbody>
                        <tr>
                            <td>Avg Classification loss</td>
                            <td>{this.props.avgClassificationLoss}</td>
                        </tr>
                        <tr>
                            <td>Examples seen</td>
                            <td>{this.props.totalExamples}</td>
                        </tr>
                        </tbody>
                    </Table>
                </div>
            </div>
        );
    }
}

export default Stats
