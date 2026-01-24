import React from 'react'
import Button from './Button'

const Main = () => {
  return (
    <>
      <div className='container'>
        <div className='p-5 text-center bg-light-dark rounded'>
          <p className='text-light lead'>This stock prediction application utilizes machine learning techniques, specifically employing Keras and an LSTM model. It integrates stock price analysis using 100-day and 200-day moving averages, which are essential indicators widely used by stock analysts to inform trading and investment decisions.</p>
          <Button text='Explore Now' class='btn-outline-info' url='/dashboard'/>
        </div>
      </div>
    </>
  )
}

export default Main