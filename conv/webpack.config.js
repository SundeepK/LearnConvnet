module.exports = {
    entry: './app/main.js',
    output: {
        path: './web',
        filename: './bundle.js',
        publicPath: "/"
    },
    watch: true,
    module: {
        loaders: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                loader: 'babel-loader'
            }
        ]
    }
};
